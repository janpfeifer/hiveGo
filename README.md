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

Anyway, GopherJS, a compiler of Go to js, made this very easy. SVG + JQuery libraries for GopherJS are not well documented ... so it took a little while to learn the ropes ... but the nice thing is: it all work on the web.

And since the Go code that runs the game can also run in compiled javascript, the whole game runs on the machine -> **TODO**: provide a link to a static version.

**TODO**: Convert the TensorFlow model to [TensorFlow.js](https://github.com/tensorflow/tfjs), so we can get good models also running on the client browser.

## Trainer

Allows playing games among AI's automatically. 

* Play new games. Optionally save the games.
* Load and rescore old games.
* Compare AIs
* Train models while playing the game.
* Can train TF models.

## Note

Thanks for Florence Poirel for the awesome drawings!

Background pictures:
* Winning pattern from [pexels.com/light-creative-abstract-colorful-134](https://www.pexels.com/photo/light-creative-abstract-colorful-134/)

