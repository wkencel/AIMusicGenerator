import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from music21 import converter, instrument, stream, note, chord



# Directory containing the MIDI files
midi_dir = "content/MidiFiles"


notes = []

# Process each MIDI file in the directory
for filename in os.listdir(midi_dir):
    if filename.endswith(".midi"):
        file = converter.parse(os.path.join(midi_dir, filename))

        # Find all the notes and chords in the MIDI file
        try:
            # If the MIDI file has instrument parts
            s2 = file.parts.stream()
            notes_to_parse = s2[0].recurse()
        except:
            # If the MIDI file only has notes (
            # no chords or instrument parts)
            notes_to_parse = file.flat.notes

        # Extract pitch and duration information from notes and chords
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in 
                element.normalOrder))

# Print the number of notes and some example notes
print("Total notes:", len(notes))
print("Example notes:", notes[:10])

# Create a dictionary to map unique notes to integers
unique_notes = sorted(set(notes))
note_to_int = {note: i for i, note in 
enumerate(unique_notes)}

# Convert the notes to numerical sequences
sequence_length = 100  # Length of each input sequence
input_sequences = []
output_sequences = []

# Generate input/output sequences
for i in range(0, len(notes) - sequence_length, 1):
    # Extract the input sequence
    input_sequence = notes[i:i + sequence_length]
    input_sequences.append([note_to_int[note] for 
    note in input_sequence])

    # Extract the output sequence
    output_sequence = notes[i + sequence_length]
    output_sequences.append(note_to_int[output_sequence])

# Reshape and normalize the input sequences
num_sequences = len(input_sequences)
num_unique_notes = len(unique_notes)

# Reshape the input sequences
X = np.reshape(input_sequences, (num_sequences, sequence_length, 1))
# Normalize the input sequences
X = X / float(num_unique_notes)

# One-hot encode the output sequences
y = to_categorical(output_sequences)

# Define the RNN model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), 
return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# # Step 4: Train the model
# model.fit(X, y, batch_size=64, epochs=100)


# # Generate new music
# def generate_music(model, seed_sequence, length):
#     generated_sequence = seed_sequence.copy()

#     for _ in range(length):
#         input_sequence = np.array(generated_sequence)
#         input_sequence = np.reshape(input_sequence, (1, len(input_sequence), 1))
#         input_sequence = input_sequence / float(num_unique_notes)  # Normalize input sequence

#         predictions = model.predict(input_sequence)[0]
#         new_note = np.random.choice(range(len(predictions)), p=predictions)
#         generated_sequence.append(new_note)
#         generated_sequence = generated_sequence[1:]

#     return generated_sequence

# # Set the seed sequence and length of the generated music
# seed_sequence = input_sequences[0]   # Replace with your own seed sequence
# generated_length = 100  # Replace with the desired length of the generated music

# generated_music = generate_music(model, seed_sequence, generated_length)
# generated_music

# # Reverse the mapping from notes to integers
# int_to_note = {i: note for note, i in note_to_int.items()}

# # Create a stream to hold the generated notes/chords
# output_stream = stream.Stream()

# # Convert the output from the model into notes/chords
# for pattern in generated_music:
#     # pattern is a number, so we convert it back to a note/chord string
#     pattern = int_to_note[pattern]

#     # If the pattern is a chord
#     if ('.' in pattern) or pattern.isdigit():
#         notes_in_chord = pattern.split('.')
#         notes = []
#         for current_note in notes_in_chord:
#             new_note = note.Note(int(current_note))
#             new_note.storedInstrument = instrument.Piano()
#             notes.append(new_note)
#         new_chord = chord.Chord(notes)
#         output_stream.append(new_chord)
#     # If the pattern is a note
#     else:
#         new_note = note.Note(pattern)
#         new_note.storedInstrument = instrument.Piano()
#         output_stream.append(new_note)

# # Write the stream to a MIDI file
# output_stream.write('midi', fp='generated_music2.mid')