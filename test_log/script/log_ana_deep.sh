#!/bin/bash
path="../"
client="deep_client_lan"
server="deep_server_lan"

grep -i "ms" "$path$client" | grep "sanns_demo" | awk '{printf " %.2f \n", $(NF-1)}' > "time.txt"

grep -i "comm:" "$path$client"| awk '{printf " %.2f \n", $(NF-1)}' > "comm.txt"

grep -i "comm" "$path$server" | grep "sanns_demo" | awk '{printf " %.2f \\\\\n", $(NF-1)}' > "recv.txt"


client="deep_client"
# server="server_wan"

grep -i "ms" "$path$client" | grep "sanns_demo" | awk '{printf "%.2f \n",$(NF-1)}' > "wan-time.txt"


paste -d '&' "time.txt" "wan-time.txt" "comm.txt" "recv.txt" 