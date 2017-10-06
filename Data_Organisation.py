import numpy as np
from mido import MidiFile  # Midi reading program


def prep_data(file_name, shape_organiser, type_count,
              decimal_check=10 * 2):  # Convert midi data into binary (true, false) statements

    mid = MidiFile(file_name)  # Change midi data into messages
    notes = []
    time = float(0)  # Floats keep us afloat
    prev = float(0)  # Floats keep us afloat

    for msg in mid:  # For message in the midi file
        # This time is in seconds, not ticks (musical ticks)
        time += msg.time
        # Only interested in the notes
        if not msg.is_meta:
            # Only interested in the piano channel (channel 1)
            if msg.channel == 0:
                if msg.type == 'note_on':
                    # Only interested in the "on" notes
                    # Note in "list" form to train on
                    note = msg.bytes()
                    # Only interested in the note (E.G. A, Bb, C#) and velocity (how long the note is).
                    # Note message is in the form of [type, note, velocity]
                    note = note[1:3]
                    note.append(time - prev)
                    prev = time  # Get each note's timing
                    notes.append(note)  # Append to the grand list of notes

    for note in notes:  # Change the time value
        note[0] = note[0]  # Stay the same
        note[1] = note[1]  # Stay the same
        note[2] = int(  # Must be an integer for binary translation.
            round(note[2] * decimal_check))  # So that we can get a precise integer that MAY equal to a decimal later.
        # NOTE: THE HIGHER THE DECIMAL CHECK, THE LONGER THE MODEL HAS TO TRAIN
    notes = np.array(notes).reshape(-1, shape_organiser, type_count)
    print notes
    X = np.zeros((-1, shape_organiser, type_count), dtype=np.bool)
    Y = np.zeros((-1, type_count), dtype=np.bool)
    # for note in notes:
    #

    return X, Y  # Important variables initialed.
