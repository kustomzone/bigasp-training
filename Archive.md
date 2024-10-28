
Archive:  https://old.reddit.com/r/StableDiffusion/comments/1dbasvx/the_gory_details_of_finetuning_sdxl_for_30m/

<hr>

There's lots of details on how to train SDXL loras, but details on how the big SDXL finetunes were trained is scarce to say the least. I recently released a big SDXL finetune. 1.5M images, 30M training samples, 5 days on an 8xH100. So, I'm sharing all the training details here to help the community.

## Finetuning SDXL

bigASP was trained on about 1,440,000 photos, all with resolutions larger than their respective aspect ratio bucket. Each image is about 1MB on disk, making the dataset about 1TB per million images.

Every image goes through: a quality model to rate it from 0 to 9; JoyTag to tag it; OWLv2 with the prompt "a watermark" to detect watermarks in the images. I found OWLv2 to perform better than even a finetuned vision model, and it has the added benefit of providing bounding boxes for the watermarks. Accuracy is about 92%. While it wasn't done for this version, it's possible in the future that the bounding boxes could be used to do "loss masking" during training, which basically hides the watermarks from SD. For now, if a watermark is detect, a "watermark" tag is included in the training prompt.

Images with a score of 0 are dropped entirely. I did a lot of work specifically training the scoring model to put certain images down in this score bracket. You'd be surprised at how much junk comes through in datasets, and even a hint of them can really throw off training. Thumbnails, video preview images, ads, etc.

bigASP uses the same aspect ratios buckets that SDXL's paper defines. All images are bucketed into the bucket they best fit in while not being smaller than any dimension of that bucket when scaled down. So after scaling, images get randomly cropped. The original resolution and crop data is recorded alongside the VAE encoded image on disk for conditioning SDXL, and finally the latent is gzipped. I found gzip to provide a nice 30% space savings. This reduces the training dataset down to about 100GB per million images.

Training was done using a custom training script based off the diffusers library. I used a custom training script so that I could fully understand all the inner mechanics and implement any tweaks I wanted. Plus I had my training scripts from SD1.5 training, so it wasn't a huge leap. The downside is that a lot of time had to be spent debugging subtle issues that cropped up after several bugged runs. Those are all expensive mistakes. But, for me, mistakes are the cost of learning.

I think the training prompts are really important to the performance of the final model in actual usage. The custom Dataset class is responsible for doing a lot of heavy lifting when it comes to generating the training prompts. People prompt with everything from short prompts to long prompts, to prompts with all kinds of commas, underscores, typos, etc.

I pulled a large sample of AI images that included prompts to analyze the statistics of typical user prompts. The distribution of prompt length followed a mostly normal distribution, with a mean of 32 tags and a std of 19.8. So my Dataset class reflects this. For every training sample, it picks a random integer in this distribution to determine how many tags it should use for this training sample. It shuffles the tags on the image and then truncates them to that number.

This means that during training the model sees everything from just "1girl" to a huge 224 token prompt. And thus, hopefully, learns to fill in the details for the user.

Certain tags, like watermark, are given priority and always included if present, so the model learns those tags strongly. This also has the side effect of conditioning the model to not generate watermarks unless asked during inference.

The tag alias list from danbooru is used to randomly mutate tags to synonyms so that bigASP understands all the different ways people might refer to a concept. Hopefully.

And, of course, the score tags. Just like Pony XL, bigASP encodes the score of a training sample as a range of tags of the form "score_X" and "score_X_up". However, to avoid the issues Pony XL ran into (shoulders of giants), only a random number of score tags are included in the training prompt. It includes between 1 and 3 randomly selected score tags that are applicable to the image. That way the model doesn't require "score_8, score_7, score_6, score_5..." in the prompt to work correctly. It's already used to just a single, or a couple score tags being present.

10% of the time the prompt is dropped completely, being set to an empty string. UCG, you know the deal. N.B.!!! I noticed in Stability's training scripts, and even HuggingFace's scripts, that instead of setting the prompt to an empty string, they set it to "zero" in the embedded space. This is different from how SD1.5 was trained. And it's different from how most of the SD front-ends do inference on SD. My theory is that it can actually be a big problem if SDXL is trained with "zero" dropping instead of empty prompt dropping. That means that during inference, if you use an empty prompt, you're telling the model to move away not from the "average image", but away from only images that happened to have no caption during training. That doesn't sound right. So for bigASP I opt to train with empty prompt dropping.

Additionally, Stability's training scripts include dropping of SDXL's other conditionings: original_size, crop, and target_size. I didn't see this behavior present in kohyaa's scripts, so I didn't use it. I'm not entirely sure what benefit it would provide.

I made sure that during training, the model gets a variety of batched prompt lengths. What I mean is, the prompts themselves for each training sample are certainly different lengths, but they all have to be padded to the longest example in a batch. So it's important to ensure that the model still sees a variety of lengths even after batching, otherwise it might overfit to a specific range of prompt lengths. A quick Python Notebook to scan the training batches helped to verify a good distribution: 25% of batches were 225 tokens, 66% were 150, and 9% were 75 tokens. Though in future runs I might try to balance this more.

The rest of the training process is fairly standard. I found min-snr loss to work best in my experiments. Pure fp16 training did not work for me, so I had to resort to mixed precision with the model in fp32. Since the latents are already encoded, the VAE doesn't need to be loaded, saving precious memory. For generating sample images during training, I use a separate machine which grabs the saved checkpoints and generates the sample images. Again, that saves memory and compute on the training machine.

The final run uses an effective batch size of 2048, no EMA, no offset noise, PyTorch's AMP with just float16 (not bfloat16), 1e-4 learning rate, AdamW, min-snr loss, 0.1 weight decay, cosine annealing with linear warmup for 100,000 training samples, 10% UCG rate, text encoder 1 training is enabled, text encoded 2 is kept frozen, min_snr_gamma=5, PyTorch GradScaler with an initial scaling of 65k, 0.9 beta1, 0.999 beta2, 1e-8 eps. Everything is initialized from SDXL 1.0.

A validation dataset of 2048 images is used. Validation is performed every 50,000 samples to ensure that the model is not overfitting and to help guide hyperparameter selection. To help compare runs with different loss functions, validation is always performed with the basic loss function, even if training is using e.g. min-snr. And a checkpoint is saved every 500,000 samples. I find that it's really only helpful to look at sample images every million steps, so that process is run on every other checkpoint.

A stable training loss is also logged (I use Wandb to monitor my runs). Stable training loss is calculated at the same time as validation loss (one after the other). It's basically like a validation pass, except instead of using the validation dataset, it uses the first 2048 images from the training dataset, and uses a fixed seed. This provides a, well, stable training loss. SD's training loss is incredibly noisy, so this metric provides a much better gauge of how training loss is progressing.

The batch size I use is quite large compared to the few values I've seen online for finetuning runs. But it's informed by my experience with training other models. Large batch size wins in the long run, but is worse in the short run, so its efficacy can be challenging to measure on small scale benchmarks. Hopefully it was a win here. Full runs on SDXL are far too expensive for much experimentation here. But one immediate benefit of a large batch size is that iteration speed is faster, since optimization and gradient sync happens less frequently.

Training was done on an 8xH100 sxm5 machine rented in the cloud. On this machine, iteration speed is about 70 images/s. That means the whole run took about 5 solid days of computing. A staggering number for a hobbyist like me. Please send hugs. I hurt.

Training being done in the cloud was a big motivator for the use of precomputed latents. Takes me about an hour to get the data over to the machine to begin training. Theoretically the code could be set up to start training immediately, as the training data is streamed in for the first pass. It takes even the 8xH100 four hours to work through a million images, so data can be streamed faster than it's training. That way the machine isn't sitting idle burning money.

One disadvantage of precomputed latents is, of course, the lack of regularization from varying the latents between epochs. The model still sees a very large variety of prompts between epochs, but it won't see different crops of images or variations in VAE sampling. In future runs what I might do is have my local GPUs re-encoding the latents constantly and streaming those updated latents to the cloud machine. That way the latents change every few epochs. I didn't detect any overfitting on this run, so it might not be a big deal either way.

Finally, the loss curve. I noticed a rather large variance in the validation loss between different datasets, so it'll be hard for others to compare, but for what it's worth:

https://i.imgur.com/74VQYLS.png

## Learnings and the Future

I had a lot of failed runs before this release, as mentioned earlier. Mostly bugs in the training script, like having the height and width swapped for the original_size, etc conditionings. Little details like that are not well documented, unfortunately. And a few runs to calibrate hyperparameters: trying different loss functions, optimizers, etc. Animagine's hyperparameters were the most well documented that I could find, so they were my starting point. Shout out to that team!

I didn't find any overfitting on this run, despite it being over 20 epochs of the data. That said, 30M training samples, as large as it is to me, pales in comparison to Pony XL which, as far as I understand, did roughly the same number of epochs just with 6M! images. So at least 6x the amount of training I poured into bigASP. Based on my testing of bigASP so far, it has nailed down prompt following and understands most of the tags I've thrown at it. But the undertraining is apparent in its inconsistency with overall image structure and having difficulty with more niche tags that occur less than 10k times in the training data. I would definitely expect those things to improve with more training.

Initially for encoding the latents I did "mixed-VAE" encoding. Basically, I load in several different VAEs: SDXL at fp32, SDXL at fp16, SDXL at bf16, and the fp16-fix VAE. Then each image is encoded with a random VAE from this list. The idea is to help make the UNet robust to any VAE version the end user might be using.

During training I noticed the model generating a lot of weird, high resolution patterns. It's hard to say the root cause. Could be moire patterns in the training data, since the dataset's resolution is so high. But I did use Lanczos interpolation so that should have been minimized. It could be inaccuracies in the latents, so I swapped over to just SDXL fp32 part way through training. Hard to say if that helped at all, or if any of that mattered. At this point I suspect that SDXL's VAE just isn't good enough for this task, where the majority of training images contain extreme amounts of detail. bigASP is very good at generating detailed, up close skin texture, but high frequency patterns like sheer nylon cause, I assume, the VAE to go crazy. More investigation is needed here. Or, god forbid, more training...

Of course, descriptive captions would be a nice addition in the future. That's likely to be one of my next big upgrades for future versions. JoyTag does a great job at tagging the images, so my goal is to do a lot of manual captioning to train a new LLaVa style model where the image embeddings come from both CLIP and JoyTag. The combo should help provide the LLM with both the broad generic understanding of CLIP and the detailed, uncensored tag based knowledge of JoyTag. Fingers crossed.

Finally, I want to mention the quality/aesthetic scoring model I used. I trained my own from scratch by manually rating images in a head-to-head fashion. Then I trained a model that takes as input the CLIP-B embeddings of two images and predicts the winner, based on this manual rating data. From that I could run ELO on a larger dataset to build a ranked dataset, and finally train a model that takes a single CLIP-B embedding and outputs a logit prediction across the 10 ranks.

This worked surprisingly well, given that I only rated a little over two thousand images. Definitely better for my task than the older aesthetic model that Stability uses. Blurry/etc images tended toward lower ranks, and higher quality photoshoot type photos tended towards the top.

That said, I think a lot more work could be done here. One big issue I want to avoid is having the quality model bias the Unet towards generating a specific "style" of image, like many of the big image gen models currently do. We all know that DALL-E look. So the goal of a good quality model is to ensure that it doesn't rank images based on a particular look/feel/style, but on a less biased metric of just "quality". Certainly a difficult and nebulous concept. To that end, I think my quality model could benefit from more rating data where images with very different content and styles are compared.

##Conclusion

I hope all of these details help others who might go down this painful path.
