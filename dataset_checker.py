import torchvision

cifar10_valid1 = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, )
cnt1 = [0 for _ in range(10)]
for i in range(0, 5000):
    cnt1[cifar10_valid1[i][1]] += 1
print(cnt1)

cifar10_valid2 = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, )
cnt2 = [0 for _ in range(10)]
for i in range(0, 5000):
    cnt2[cifar10_valid2[i][1]] += 1
print(cnt2)
