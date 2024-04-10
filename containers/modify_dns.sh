#!/bin/bash

# Extract the first nameserver entry
nameserver=$(grep -m 1 'nameserver' /etc/resolv.conf)

# Recreate resolv.conf with the desired content
{
    echo "search dns.podman"
    echo "$nameserver"
} > /etc/resolv.conf

# Execute the original CMD passed to the script
exec "$@"
