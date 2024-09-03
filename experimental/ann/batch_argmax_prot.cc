#include "batch_argmax_prot.h"

#include "bitwidth_change_prot.h"

#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
namespace sanns {
// static uint64_t send_rounds = 0;
// static uint64_t send_bytes = 0;

BatchArgmaxProtocol::BatchArgmaxProtocol(
    const std::shared_ptr<spu::KernelEvalContext> &ctx, size_t compare_radix)
    : compare_radix_(compare_radix), ctx_(ctx) {
  is_sender_ = ctx->lctx()->Rank();
}

spu::NdArrayRef BatchArgmaxProtocol::Compute(const spu::NdArrayRef &inp,
                                             const int64_t bitwidth,
                                             const int64_t shift,
                                             const size_t batch_size,
                                             const size_t max_size) {
  SPU_ENFORCE(inp.ndim() == 2);
  SPU_ENFORCE(inp.dim(0) == batch_size && inp.dim(1) == max_size);
  int64_t level_size = max_size;
  auto tmp_res = spu::mpc::ring_rshift(inp, shift);
  auto cmp_bw = bitwidth - shift;
  while (level_size != 1) {
    // std::cout << ctx_->sctx()->lctx()->GetStats()->sent_actions.load()
    //           << std::endl;
    // std::cout << level_size << std::endl;
    int64_t drelu_size = level_size / 2;
    int64_t drelu_mod = level_size % 2;
    int64_t cmp_size = drelu_mod + drelu_size;
    spu::Shape drelu_shape({static_cast<int64_t>(cmp_size * batch_size)});
    // TODO(ljy): change to slice
    spu::NdArrayRef a_value(tmp_res.eltype(), drelu_shape);
    spu::NdArrayRef b_value(tmp_res.eltype(), drelu_shape);
    spu::pforeach(0, batch_size, [&](int64_t i) {
      std::memcpy(&a_value.at(i * cmp_size), &tmp_res.at(i * level_size),
                  cmp_size * tmp_res.elsize());
      std::memcpy(&b_value.at(i * cmp_size),
                  &tmp_res.at(i * level_size + drelu_size),
                  cmp_size * tmp_res.elsize());
    });
    spu::mpc::ring_sub_(a_value, b_value);
    spu::mpc::ring_bitmask_(a_value, 0, cmp_bw);
    // TODO(ljy): Add Index
    // spu::NdArrayRef sub_index(inp.eltype(), max_sub_size);
    auto select_bits = DReLU(a_value, cmp_bw);
    auto res = Select(select_bits, a_value);
    spu::mpc::ring_add_(res, b_value);
    spu::mpc::ring_bitmask_(res, 0, cmp_bw);
    tmp_res = res;
    level_size = level_size / 2 + level_size % 2;
  }
  return tmp_res;
}

// input bitwidth : bitwidth
// outptu bitwidth : outbitwidth
spu::NdArrayRef BatchArgmaxProtocol::TruncateAndReduce(
    const spu::NdArrayRef &inp, int64_t bitwidth, int64_t shift) {
  // spu::NdArrayRef trun_value;
  auto trun_value = spu::mpc::cheetah::TiledDispatchOTFunc(
      ctx_.get(), inp,
      [&](const spu::NdArrayRef &input,
          const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols> &base_ot) {
        BitWidthChangeProtocol trun_prot(base_ot);
        return trun_prot.TrunReduceCompute(input, bitwidth, shift);
      });
  return trun_value;
}

std::vector<spu::NdArrayRef> BatchArgmaxProtocol::ComputeWithIndex(
    const spu::NdArrayRef &inp, const spu::NdArrayRef &index,
    const int64_t bitwidth, const int64_t shift, const size_t batch_size,
    const size_t max_size) {
  SPU_ENFORCE(inp.ndim() == 2);
  SPU_ENFORCE(inp.dim(0) == batch_size && inp.dim(1) == max_size);
  int64_t level_size = max_size;
  // auto tmp_res = spu::mpc::ring_rshift(inp, shift);
  // tmp_res = Trun
  auto tmp_res = TruncateAndReduce(inp, bitwidth, shift);
  auto tmp_idx = index;
  auto cmp_bw = bitwidth - shift;

  while (level_size != 1) {
    // std::cout << ctx_->sctx()->lctx()->GetStats()->sent_actions.load()
    //           << std::endl;
    // std::cout << level_size << std::endl;
    int64_t drelu_size = level_size / 2;
    int64_t drelu_mod = level_size % 2;
    int64_t cmp_size = drelu_mod + drelu_size;
    spu::Shape drelu_shape({static_cast<int64_t>(cmp_size * batch_size)});
    // TODO(ljy): change to slice
    spu::NdArrayRef a_value(tmp_idx.eltype(), drelu_shape);
    spu::NdArrayRef b_value(tmp_idx.eltype(), drelu_shape);
    spu::NdArrayRef a_index(tmp_idx.eltype(), drelu_shape);
    spu::NdArrayRef b_index(tmp_idx.eltype(), drelu_shape);

    spu::pforeach(0, batch_size, [&](int64_t i) {
      std::memcpy(&a_value.at(i * cmp_size), &tmp_res.at(i * level_size),
                  cmp_size * tmp_res.elsize());

      std::memcpy(&b_value.at(i * cmp_size),
                  &tmp_res.at(i * level_size + drelu_size),
                  cmp_size * tmp_res.elsize());

      std::memcpy(&a_index.at(i * cmp_size), &tmp_idx.at(i * level_size),
                  cmp_size * tmp_idx.elsize());

      std::memcpy(&b_index.at(i * cmp_size),
                  &tmp_idx.at(i * level_size + drelu_size),
                  cmp_size * tmp_idx.elsize());
    });
    spu::mpc::ring_sub_(a_value, b_value);
    spu::mpc::ring_bitmask_(a_value, 0, cmp_bw);

    auto select_bits = DReLU(a_value, cmp_bw);
    // TODO(ljy): add index bitwidth (index bw may not equal with value bw)
    spu::NdArrayRef select_bits_double(
        select_bits.eltype(),
        {static_cast<int64_t>(batch_size * cmp_size * 2)});

    spu::mpc::ring_sub_(a_index, b_index);
    spu::mpc::ring_bitmask_(a_index, 0, bitwidth);

    spu::NdArrayRef data(a_value.eltype(),
                         {static_cast<int64_t>(batch_size * cmp_size * 2)});
    std::memcpy(&select_bits_double.at(0), &select_bits.at(0),
                batch_size * cmp_size * select_bits.elsize());

    std::memcpy(&select_bits_double.at(batch_size * cmp_size),
                &select_bits.at(0),
                batch_size * cmp_size * select_bits.elsize());

    std::memcpy(&data.at(0), &a_value.at(0),
                batch_size * cmp_size * a_value.elsize());

    std::memcpy(&data.at(batch_size * cmp_size), &a_index.at(0),
                batch_size * cmp_size * a_index.elsize());

    auto merge_res = Select(select_bits_double, data);

    spu::NdArrayRef res(a_value.eltype(), drelu_shape);
    spu::NdArrayRef idx(a_value.eltype(), drelu_shape);

    std::memcpy(&res.at(0), &merge_res.at(0),
                batch_size * cmp_size * a_value.elsize());

    std::memcpy(&idx.at(0), &merge_res.at(batch_size * cmp_size),
                batch_size * cmp_size * idx.elsize());

    spu::mpc::ring_add_(res, b_value);
    spu::mpc::ring_bitmask_(res, 0, cmp_bw);

    spu::mpc::ring_add_(idx, b_index);
    spu::mpc::ring_bitmask_(idx, 0, bitwidth);

    tmp_res = res;
    tmp_idx = idx;
    level_size = level_size / 2 + level_size % 2;
  }
  return {tmp_res, tmp_idx};
}

spu::NdArrayRef BatchArgmaxProtocol::Select(spu::NdArrayRef &select_bits,
                                            spu::NdArrayRef &a) {
  // TODO: select value and index together
  // TODO: sele
  SPU_ENFORCE_EQ(a.shape(), select_bits.shape());
  const int64_t numel = select_bits.numel();
  if (numel == 0) {
    return spu::NdArrayRef(a.eltype(), a.shape());
  }
  return spu::mpc::cheetah::TiledDispatchOTFunc(
             ctx_.get(), a, select_bits,
             [&](const spu::NdArrayRef &input0, const spu::NdArrayRef &input1,
                 const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>
                     &base_ot) { return base_ot->Multiplexer(input0, input1); })
      .as(a.eltype());
}

spu::NdArrayRef BatchArgmaxProtocol::DReLU(spu::NdArrayRef &inp,
                                           int64_t bitwidth) {
  // TODO: use lut to accelerate the Drelu( communication cost higher but round
  // lower)
  auto numel = inp.numel();

  auto field = inp.eltype().as<spu::Ring2k>()->field();
  if (inp.numel() == 0) {
    spu::NdArrayRef res(inp.eltype(), inp.shape());
    return res.as(spu::makeType<spu::mpc::cheetah::BShrTy>(field, 1));
  }

  const size_t shft = bitwidth - 1;
  // delete dispatch All Fields code to simplify the code

  // TODO: sanns only need use FM32;
  using namespace spu;
  return DISPATCH_ALL_FIELDS(field, "_", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const u2k mask = (static_cast<u2k>(1) << shft) - 1;
    NdArrayRef adjusted = spu::mpc::ring_zeros(field, inp.shape());
    auto xinp = NdArrayView<const u2k>(inp);
    auto xadj = NdArrayView<u2k>(adjusted);
    if (is_sender_ == 0) {
      pforeach(0, numel, [&](int64_t i) { xadj[i] = xinp[i] & mask; });
    } else {
      pforeach(0, numel, [&](int64_t i) { xadj[i] = (mask - xinp[i]) & mask; });
    }

    auto carry_bit =
        spu::mpc::cheetah::TiledDispatchOTFunc(
            ctx_.get(), adjusted,
            [&](const NdArrayRef &input,
                const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>
                    &base_ot) {
              spu::mpc::cheetah::CompareProtocol prot(base_ot, compare_radix_);
              return prot.Compute(input, true, shft);
            })
            .as(inp.eltype());
    NdArrayView<u2k> _carry_bit(carry_bit);
    pforeach(0, numel, [&](int64_t i) { _carry_bit[i] ^= (xinp[i] >> shft); });

    return carry_bit.as(spu::makeType<spu::mpc::cheetah::BShrTy>(field, 1));
  });
}
}  // namespace sanns