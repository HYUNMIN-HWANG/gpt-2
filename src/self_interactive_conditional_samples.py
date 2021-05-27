#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    # model_name='124M',
    model_name='1558M',
    seed=None,
    # nsamples=1,
    nsamples=3, # 한 번에 출력되는 sample수 가 늘어남 
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                # print(out) 
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)


'''
out
[[   13   314   892   345   815  1282  1863   319   326   649  5581   905
   1444   705   464  9356  5338  2637   314   761   257   649  5166   286
  21726    13   632   338  2407    11  2407  7932   996   526   198   198
   5297    11 40003    11   355   356   760    13  7459  1139   339   460
  25073   852   287   262 35949   485  5588   780   339   338   366  9948
     76   284   257  8046   526   843   356   821  1654   339  1183   307
   5762   465  2042 36953   618   339   857   787   262   905    13  2329
  26471    13  1002   339  1595   470 43519    11   339   743   670   355
   9356  5338   338 32520 26494 15992   290   991   423   281  2562   640
   5291   340  1978    13   887   307  7728    11  3362    40    70  8207
    274    11   262  8737   286   262 35949   485  5588   318   326   645
    530   468  1865  5257   284  3938 13180   663 48565    13   198   198
   8585 21119 25418   198   198  7594 15105    11  2681    12  1941    12
    727 21119 25418   468  1541  9722   656   607  2597   290   318  2407
    262  3785    13   198   198     1  3347   338   635   845  5814   553
   1139  1338 46494    13   366    34   585 25418  5818   470  3421   287
   7480  2063    11   523   673   338  1464   262   976    13  1375   468
    257 14284  7947    13   921  1826   607    11   290    11   379   717
  16086    11   345   821   588    11   705  5812    11  5803  2637  1375
    338   655   262  3641   286  3241    11   475    11   287  1109    11
    661  5490   546   607   526   198   198    34   585 25418    11  5214
     11   318   257  3950  3195  9920    13  1375   338   635   257  4336
    286   366 20489  3822   553   523   428   366 20489  3822     1  7110
   7622  2406   510    13  1375  1541   468   257 12199  1628   287   262
   2499    11   475   356   821 22908   366    33 15352    25   383 21463
  27076     1   318   287   607  2003    13   198   198 17353   370  5282
   1355   257 10631  2249 15062  1154   287   366   464 14207 13984   198
    198  5189  1781   673   561    13   317 22814    51  9936 42789 27438
     14    54 49048   685    33 15352    60 27356   655  1595   470  2344
  33021   510   262   835   340 13520   510 33021    14 48035   365   685
     33 15352    60   287   366    50  1073 26730    12    35  2238   553
    262 17343  2168    13   775  1549   772  7048   326 50117   685   325
  11632 11544 10686   422   366  3163   808     1   290 19475    12   397
   1484 41406  1455   291   422   366 12442  7081     1 19152   994   416
   8484 19255 34603 13167     6    43   488   259 11661    60   318   257
   4574  2502   618   339   338   850 33532   284   465  7954  6478    11
   4216   346   993   685 26141    88 11955   263    60  7043 22091  7456
    422   366 22697   418   553   366    35  1347    51  2040     1  9193
     13    15    13   357 23937    11   339  3073 17855   588 11336 29715
   2014   198   198  2202   262   584  1021    11  8393  6951  3700 40929
    357 11696 44311    11   509  2086   263 47793     8   290  4916  7311
   2452 17608   685    50 15918  6206 10424    11   718 39285  5890    11
    718  9506  1024  5379    11   642 19534    33]]

======================================== SAMPLE 1 ========================================
. I think you should come along on that new television show called 'The Doctor Who.' I need a new pair of lungs. It's quite, quite wonderful though." 

'''