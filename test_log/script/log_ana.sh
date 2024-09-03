#!/bin/bash
path="../"
client="client1.txt"
server="server1.txt"

grep -i "ms" "$path$client" | grep "sanns_demo" | awk '{printf "%s & %.2f \n", $5,$(NF-1)}' > "time.txt"

grep -i "comm:" "$path$client"| awk '{printf " %.2f \n", $(NF-1)}' > "comm.txt"

grep -i "comm" "$path$server" | grep "sanns_demo" | awk '{printf " %.2f \n", $(NF-1)}' > "recv.txt"


client="client1_wan.txt"
# server="server1_wan.txt"

grep -i "ms" "$path$client" | grep "sanns_demo" | awk '{printf "%.2f \n",$(NF-1)}' > "wan-time.txt"


paste -d '&' "time.txt" "wan-time.txt" "comm.txt" "recv.txt" deep2.txt 