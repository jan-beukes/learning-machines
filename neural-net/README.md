# Neural network "library"

# TODO:
- add random preproccessing for image data

## train/view examples
```sh
odin build . -o:speed
./neural-net [train] (digits|fashion|cifar)
```

## predict digits from drawing
```sh
odin build draw -o:speed
./draw
```

## example datasets
- [Iris](archive.ics.uci.edu/dataset/53/iris)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## References
https://youtu.be/hfMk-kjRv4c

https://neuralnetworksanddeeplearning.com/index.html

https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
