<img align="left" src="images/Queen.png" alt="Queen Bee" width="64px"/>

# hiveGo <br/>

[Hive game (wikipedia)](https://en.wikipedia.org/wiki/Hive_(game)), backed by an [AlphaZero (wikipedia)](https://en.wikipedia.org/wiki/AlphaZero) 
based AI, using [GoMLX (an ML framework for Go)](https://github.com/gomlx/gomlx) .

<img align="right" src="www/hive/assets/endgame.png" alt="Ant" width="100px"/>

Hive is a small fun board game played with two people, where each one is trying to surround the other one's Queen Bee.
For a quick match, with instructions and a tutorial, 
[try a web version here (janpfeifer.github.io)](https://janpfeifer.github.io/hiveGo/www/hive/).

It is also a great platform to play around with game algorithms, like [AlphaZero (wikipedia)](https://en.wikipedia.org/wiki/AlphaZero) and demo [GoMLX](https://github.com/gomlx/gomlx),
a machine learning framework for Go.

It comes with:

## [Web Version: compiling Go to WebAssembly](https://janpfeifer.github.io/hiveGo/www/hive/)

It is slower because of WebAssembly and using a pure Go backend for the machine learning, but it is enough to play at the medium level at no wait.

<img src="https://github.com/user-attachments/assets/d6531652-bb46-4cbb-b6e1-ef9088354374" alt="image" width="800" height="600">

## Command-line UI

Tested on Linux and Windows, probably work in most platforms:

```
    go install github/janpfeifer/hiveGo/cmd/hive@latest
```

* To play vs a default AI: `hive`
* To watch AIs playing against each other: `hive -watch`
* Hotseat mode: `hive -hotseat`
* See `hive -help` for more options.

![image](https://github.com/user-attachments/assets/f67d8ad5-f047-4154-843e-4319aa55b794)

It uses the [GoMLX](https://github.com/gomlx/gomlx) Go backend  by default, so nothing else is needed to install. 
If you want the "accelerated" version, based on [OpenXLA](https://openxla.org/), including GPU support, install the XLA
libraries (see [GoMLX Installation](https://github.com/gomlx/gomlx?tab=readme-ov-file#installation)), and install **hive** command with:

```
    go install -tags xla github/janpfeifer/hiveGo/cmd/hive@latest
```


## Gnome Version

> [!WARNING]
> 🚧🛠 Currently broken 🚧🛠 <br/>

Easy to build in XWindows -- well, more or less, depending on the dependencies working out fine. Probably one could compile it in windows as well ... 

TODO: use Fyne to make it portable?

```
    go run github/janpfeifer/hiveGo/cmd/gnome-hive
```

![image](https://github.com/user-attachments/assets/87fe827c-14b8-4367-91d9-98a9be067f89)

## About The Project

This is an experimental / educative project to learn the AlphaZero algorithm and RL in general.

It was originally a 2018 project using TensorFlow, refreshed in 2025 to use [GoMLX](https://github.com/gomlx/gomlx),
a feature-full ML framework for Go.

What is working right now:

1. The command-line UI, see below.
2. AIs, using a basic set of features for the board (lots that can be improved here)
   a. AlphaBeta-Pruning ([code](https://github.com/janpfeifer/hiveGo/blob/main/internal/searchers/alphabeta/alphabeta.go)): 
      using a straight forward [FNN (Feedforward Neural Network) model](https://github.com/janpfeifer/hiveGo/blob/main/internal/ai/gomlx/fnn.go).
   b. Alpha-Zero ([code](https://github.com/janpfeifer/hiveGo/blob/main/internal/searchers/mcts/mcts.go)): 
      using [a tiny GNN (Graph Neural Network) model](https://github.com/janpfeifer/hiveGo/blob/main/internal/ai/gomlx/alphazerofnn.go).

It already beats me every time, and when I pitched it against some commercially available Hive game, both AIs won
every time.

## Trainers: [`cmd/a0-trainer`](https://github.com/janpfeifer/hiveGo/tree/main/cmd/a0-trainer) and [`cmd/ab-trainer](https://github.com/janpfeifer/hiveGo/tree/main/cmd/ab-trainer)

Allows playing games among AI's automatically. 

* Play new games. Optionally save the games.
* Load and rescore old games.
* Compare AIs
* Train models while playing the game.
* Can train TF models.
* Can distill from previous models: very handy to quickly ramp up a new model to a moderate quality model.
* Two trainers: ab-trainer (for AlphaBeta Pruning) and a0-trainer (for the AlphaZero models).

During training with larger models you may want to use GoMLX with a GPU.

## Hexagonal Convolutional model

> [!WARNING]
> 🚧🛠 Currently broken 🚧🛠 <br/>
> The GoMLX version doesn't yet implemented the convolutional model.
> But also, the better idea would be to do a transformer based model.

The latest, more fancy model, uses the full board as input to the NN, and
runs a bunch of layers of convolution on that, with residual connections.

The convolutions for hexagonal mappings require some extra care, since the
neighbourhood kernel is different depending if one are on odd/even columns.

See experiments in [this colab](https://colab.research.google.com/drive/1r4P5Uc3S5Lw3sznEVMrbF3H9HkskZH6S)

## Thanks

* Thanks for Florence Poirel for the awesome drawings!
* The project started from [earlier version in Python](https://github.com/makatony/hiveAI), but it was greatly improved since.

Background pictures:
* Winning pattern from [pexels.com/light-creative-abstract-colorful-134](https://www.pexels.com/photo/light-creative-abstract-colorful-134/)

