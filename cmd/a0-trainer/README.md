# a0-trainer 

A command line tool to orchestrate the training of Alpha-Zero based models.

It works by:

 1. Self-play a bunch of matches and collect training data.
 2. Train a copy of the model with the new data.
 3. Pitch the previous model with the newly trained one.
 4. If newly trained model is not > 10% better than previous model, discard the newly trained model
    (but not the data accumulated so far) and return to (1) and to collect more data.
 5. Else newly trained model becomes the new current model (and discard the previous one),
    checkpoint (save) it, and restart at 1.

See -help for flags.