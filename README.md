# hiveGo
Go Implementation of Hive game.

Based on [earlier version in Python](https://github.com/makatony/hiveAI), which was a bit slow to generate training data for a Reinforcement Learning AI -- future planned.

Just a toy project.

## Command line version

Hopefully this should work on linux box with Go properly installed:

```
    go get github/janpfeifer/hiveGo
    go install github/janpfeifer/hiveGo && hiveGo
```

## Gnome Version

Easy to build in XWindows -- well, more or less, depending on the dependencies working out fine. Probably one could compile it in windows as well ... 

```
    go get github/janpfeifer/hiveGo
    go install github/janpfeifer/hiveGo/gnome-hive && \
      gnome-hive -p1=ai -ai=ab,max_depth=1 --vmodule=main=1,alpha_beta_pruning=1,linear_scorer=1 --logtostderr
```

## Web Version

The Gnome version works nicely ... but asking anyone to install it is cruel. And I wouldn't want to distribute a binary -- then I would have to try to compile everything staticly.

So instead, why not a web version ... duh, I should have done this first, but I wanted to try out coding for Gnome.

Anyway, GopherJS, a compiler of Go to js, made this very easy. SVG + JQuery libraries for GopherJS are not well documented ... so it took a little while to learn the ropes ... but the nice thing is: it all work on the web. Unfortunately too slow ... this game speed (or slowness) is dominated by generating the list of moves (updating connectivity of the graph and moves for Ants), and that can't be pushed to tensorflow.js. So the fully javascript solution is not very viable for higher levels of play.

And since the Go code that runs the game can also run in compiled javascript, the whole game runs on the machine -> **TODO**: provide a link to a static version.


**TODO**: Convert the TensorFlow model to [TensorFlow.js](https://github.com/tensorflow/tfjs), so we can get good models also running on the client browser.

## Trainer

Allows playing games among AI's automatically. 

* Play new games. Optionally save the games.
* Load and rescore old games.
* Compare AIs
* Train models while playing the game.
* Can train TF models.
* Can distill from previous models: very handy to quickly ramp up a new model to a moderate quality model.

## Hexagonal Convolutional model

The latest, more fancy model, uses the full board as input to the NN, and
runs a bunch of layers of convolution on that, with residual connections.

The convolutions for hexagonal mappings require some extra care, since the
neighbourhood kernel is different depending if one are on odd/even columns.

See experiments in [this colab](https://colab.research.google.com/drive/1r4P5Uc3S5Lw3sznEVMrbF3H9HkskZH6S)

## Note

Thanks for Florence Poirel for the awesome drawings!

Background pictures:
* Winning pattern from [pexels.com/light-creative-abstract-colorful-134](https://www.pexels.com/photo/light-creative-abstract-colorful-134/)

