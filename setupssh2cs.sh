#!/bin/bash

sudo mkdir -p /var/run/sshd && sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && sudo sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config && sudo service ssh restart && echo "✅ sshd running for GitHub Codespaces"
