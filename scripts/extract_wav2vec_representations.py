import fairseq
from transformers import Wav2Vec2Model
import torchaudio
import os
import csv
import time
from representations.representation_extraction import extract_features


start_time = time.time()

dataset = 'PC-GITA'                         # 'English' or 'PC-GITA'

model_label = 'w2v1'                        # 'w2v1' or 'w2v2'
representation_level = 'FeatureExtractor'   # 'FeatureExtractor' or 'FeatureAggregator' or 'LastHiddenState' or 'TransformerLayer'
transformer_layer = False                   # only if representation_level == 'TransformerLayer'
pooling = 'mean'                            # 'mean' or 'max'


if model_label == 'w2v1':
    model_path = 'models/wav2vec_large.pt'
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = model[0].eval()

if model_label == 'w2v2':
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = model.eval()

    print("\nModel Architecture:\n")
    for name, module in model.named_modules():
        print(name, '->', module)


def get_speech_representations(folder_path, writer, dataset, speaker_class, speech_task, model, representation_level, pooling):
    audio_paths = sorted(
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path)
        if f.endswith(".wav"))

    ids = set()
    
    for audio_path in audio_paths:
        start_audio_time = time.time()
        
        original_audio, original_sample_rate = torchaudio.load(audio_path)

        if dataset == 'English':
            id = os.path.splitext(os.path.basename(audio_path))[0]
        if dataset == 'PC-GITA':
            id = os.path.splitext(os.path.basename(audio_path))[0].split('_')[0].split('-')[0].split('a')[0]

        if id not in ids:
            # audio length
            audio_length = original_audio.size(1) / original_sample_rate

            # resampling
            if original_sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=16000)
                audio_input = resampler(original_audio)
            else:
                audio_input = original_audio

            # features extraction
            final_repre = extract_features(model, model_label, representation_level, transformer_layer, pooling, audio_input)

            end_audio_time = time.time()
            RunTime = end_audio_time - start_audio_time

            # saving a row to a file
            row = {
                'id': id,
                'speaker_class': speaker_class,
                'speech_task': speech_task,
                'audio_length': audio_length,
                'RunTime': RunTime,
            }

            row.update({
                f'Feature{i + 1}': value
                for i, value in enumerate(final_repre)
            })
            
            if writer is None:
                NumOfFeatures = len(final_repre)
                header = ['id', 'speaker_class', 'speech_task', 'audio_length', 'RunTime'] + [f'Feature{i + 1}' for i in range(NumOfFeatures)]
                writer = csv.DictWriter(csv_file, fieldnames=header)
                writer.writeheader()

            writer.writerow(row)

            ids.add(id)
    return


# new file with header
FileName = f"{model_label}_{representation_level}{f'[{transformer_layer}]' if transformer_layer else ''}_{pooling}Repre_{dataset}.csv"
    
with open(FileName, "w", newline="") as csv_file:
    writer = None


    # DATA:

    # Read Text
    get_speech_representations("data/"+dataset+"/ReadText/HC", writer, dataset, 'HC', 'ReadText', model, representation_level, pooling)
    get_speech_representations("data/"+dataset+"/ReadText/PD", writer, dataset, 'PD', 'ReadText', model, representation_level, pooling)

    if dataset == 'PC-GITA':
        # Monologue
        get_speech_representations("data/"+dataset+"/monologue/HC", writer, dataset, 'HC', 'Monologue', model, representation_level, pooling)
        get_speech_representations("data/"+dataset+"/monologue/PD", writer, dataset, 'PD', 'Monologue', model, representation_level, pooling)
    if dataset == 'English':
        # Spontaneous Dialogue
        get_speech_representations("data/"+dataset+"/SpontaneousDialogue/HC", writer, dataset, 'HC', 'SpontaneousDialogue', model, representation_level, pooling)
        get_speech_representations("data/"+dataset+"/SpontaneousDialogue/PD", writer, dataset, 'PD', 'SpontaneousDialogue', model, representation_level, pooling)

    # Vowel A
    get_speech_representations("data/"+dataset+"/VowelA/HC", writer, dataset, 'HC', 'VowelA', model, representation_level, pooling)
    get_speech_representations("data/"+dataset+"/VowelA/PD", writer, dataset, 'PD', 'VowelA', model, representation_level, pooling)


end_time = time.time()

run_time = end_time - start_time
print(f"Run time: {run_time} second")


