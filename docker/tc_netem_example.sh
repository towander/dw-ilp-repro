#!/bin/bash
# Example: emulate 50ms delay and 1% packet loss
tc qdisc add dev eth0 root netem delay 50ms loss 1%
