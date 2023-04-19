datasets=("CIFAR10" "CIFAR100" "MNIST" "FashionMNIST")

for i in "${!datasets[@]}"
do
  for j in "${!datasets[@]}"
  do
    if [[ "$i" != "$j" && "${datasets[i]}" != "${datasets[j]}" ]]
    then
      echo "Selected pair: ${datasets[i]}, ${datasets[j]}"
      sudo python3 main.py --in_dataset "${datasets[i]}" --out_dataset "${datasets[j]}" --batch_size 128
    fi
  done
done
