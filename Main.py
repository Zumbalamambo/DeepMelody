from mido import MidiFile

from Data_Organisation import prep_data
from Model import music_model

input_of_midi = raw_input("Please enter the file's name: ")
type_count = 3
# INITIALIZE VARIABLES
mid = MidiFile(input_of_midi)
decimal_check = 10 ** 2
shape_organiser = 3

X, Y = prep_data(input_of_midi,
                 shape_organiser,
                 type_count,
                 decimal_check=decimal_check)  # All important variables organised. Nice.
# The X and Y variables are temporary. We are after the train_seed and test_seed variables.
# They will be used for predictions.

model = music_model(X, Y, shape_organiser, number_of_epochs=128)
