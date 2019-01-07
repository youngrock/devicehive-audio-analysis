import argparse
import os
import numpy as np
import librosa
from moviepy import editor
# custom video utils
import frames

parser = argparse.ArgumentParser(description='Read audio/video file and predict sound labels')
parser.add_argument('input_file', type=str, help='File to read and process')
parser.add_argument('-f','--feature_length', type=float, help = 'Length of sound segment for feature', default=2.)
parser.add_argument('-t','--time_step', type=float, help='Time step between predictions',default=2.)
parser.add_argument('-n','--num_pred', type=int, help='Number of prediction labels to display',default=3)
parser.add_argument('-o','--output',action='store_true',help='Optional video ouput file with prediction rendering')

def predict_sound(input_file,time_step=2.,feature_length=2.,num_pred=3):
    """
    Predict sound labels for given wav or mp4 files
        input_file: name of the audio/video input file
        time_step: size of time steps for atomic feature
        num_pred: number of prediction labels to output
    """
    # use librosa load to ensure datatype and sample rate
    #sr, data = wavfile.read(wav_file)
    # Current trained model requires wav to be mono 16K sample rate
    data, sr = librosa.load(input_file,sr=16000,mono=True)
    # NOTE: librosa always returns float32 type, convert it to int16
    data = np.int16(data*32768)

    #total_duration = len(data)/sr
    # local import to reduce start-up time
    from audio.processor import WavProcessor, format_predictions

    prediction_seq = []
    with WavProcessor() as proc:
        # loop over the sound segments
        # segment sound by clips of time-step length
        inc_step = int(sr*time_step)
        tmp_steps = list(range(0,len(data)+1,inc_step))
        feature_step = int(sr*feature_length)
        #if tmp[-1] != len(data):
        #    tmp.append(len(data))
    
        for a,b in zip(tmp_steps[:-1],tmp_steps[1:]):
            d = data[a:a+feature_step]

            if all(d==0):
                # if all data entry is zero, this is silence
                predictions = [('Silence', 1.0)]
            else:
                predictions = proc.get_predictions(sr, d)

            # format output
            # sort labels on confidence
            # show first 5 labels
            # save to csv file (or json)
            # a denotes the start time
            start_time = a/sr
            end_time = b/sr
            prediction_seq.append(((start_time,end_time),predictions[:num_pred]))
            prediction_str = format_predictions(predictions,num_pred)
            print("({0},{1}): {2}".format(start_time,end_time,prediction_str))

    return prediction_seq


if __name__ == '__main__':
    args = parser.parse_args()
    prediction = predict_sound(input_file=args.input_file,
            time_step=args.time_step,
            feature_length=args.feature_length,
            num_pred=args.num_pred)
    # getting prediction takes long time - loading model especially
    # so try to use saved prediction if you want to debug the below code
    #with open('prediction.pickle','rb') as fp:
    #    import pickle
    #    prediction = pickle.load(fp)

    # Optionally, rendering prediction output to input video for visualization

    del(prediction[-1])     # last segment may have wrong length, so just delete it.

    if args.output is True:
        mclip = editor.VideoFileClip(args.input_file)
        oclips = []
        for p in prediction:
            (from_t,to_t) = p[0]
            labels = ''
            for l in p[1]:
                labels += '{:8}: {:.2}\n'.format(l[0][:7],l[1])
            sub_mclip = mclip.subclip(from_t,to_t)
            txtclip = editor.TextClip(labels,fontsize=20,color='white',align='West',stroke_width=2)
            cvc = editor.CompositeVideoClip([sub_mclip,txtclip.set_position((0.2,0.8),relative=True)])
            #cvc = editor.CompositeVideoClip([sub_mclip,txtclip.set_position('center','South')])
            oclips.append(cvc.set_duration(sub_mclip.duration))
        final_clip = editor.concatenate_videoclips(oclips)
        base = os.path.basename(args.input_file).split('.')
        output_file_name = os.path.join(os.path.dirname(args.input_file),base[0]+'.sound.'+base[1])
        final_clip.write_videofile(output_file_name)


