#!/bin/bash
# Run this on nodes that have a GPU
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl label node $NODE node-type=gpu --overwrite
kubectl taint node $NODE gpu=true:NoSchedule --overwrite
echo "Node $NODE labeled as GPU node"
