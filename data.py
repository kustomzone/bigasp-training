from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler
from transformers import CLIPTokenizer
import gzip
import struct
import random
import json
from collections import defaultdict
from typing import NamedTuple, Optional, Iterator
import torch.distributed as dist
import math
import dataclasses
from tqdm import tqdm


@dataclasses.dataclass
class Bucket:
	resolution: tuple[int, int]
	aspect: float
	n_chunks: int
	images: list[int]



#Bucket = NamedTuple('Bucket', [('resolution', tuple[int, int]), ('aspect', float), ('images', list[int])])


# These tags will always be included if they are present in the tag string
IMPORTANT_TAGS = set(['watermark'])


class ImageDatasetPrecoded(Dataset):
	"""
	Precomputed latents
	n_tags_mean and n_tags_std are based on measurements done on gen databases.
	"""
	def __init__(self, data, tokenizer: CLIPTokenizer, tokenizer_2: CLIPTokenizer, datapath: Path, n_tags_mean: float = 32, n_tags_std: float = 19.8, tag_percent: float = 0.1):
		super().__init__()
		self.data = data
		self.tokenizer = tokenizer
		self.tokenizer_2 = tokenizer_2
		self.datapath = Path(datapath)
		self.n_tags_mean = n_tags_mean
		self.n_tags_std = n_tags_std
		self.tag_percent = tag_percent

		# Read aliases
		# read_tag_aliases goes from aliased tags back to a canonical tag
		# So we need to invert it
		tag_aliases = read_tag_aliases()
		inv_aliases = defaultdict(list)
		for k, v in tag_aliases.items():
			inv_aliases[v].append(k)
		
		self.tag_aliases = inv_aliases
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx: tuple[tuple[int, int], int, int]):
		resolution, index, epoch = idx
		row = self.data[index]
		image_hash = row['image_hash']
		image_caption: str = row['caption']
		image_tags: str = row['tag_string']
		image_score: int = row['score']
		image_n_tokens: int = row['n_tokens']
		rng = random.Random(hash((index, epoch)))

		precomputed_path = self.datapath / image_hash[:2] / image_hash[2:4] / f"{image_hash}.bin.gz"
		with gzip.open(precomputed_path, "rb") as f:
			original_width, original_height, crop_x, crop_y, latent_width, latent_height = struct.unpack("<IIIIII", f.read(24))
			assert latent_width == row['latent_width'] and latent_height == row['latent_height'], f"Expected latent size {row['latent_width']}x{row['latent_height']}, got {latent_width}x{latent_height}"
			assert (latent_width, latent_height) == resolution, f"Expected resolution {resolution}, got {(latent_width, latent_height)}"
			data = f.read()
		
		latent = torch.frombuffer(bytearray(data), dtype=torch.float16).view(4, latent_width, latent_height)

		# Build prompt
		if rng.random() < self.tag_percent:
			# We match the caption length, so the n_chunk bucketing works
			# On average a tag costs 2.5 tokens
			n_tags = int(image_n_tokens / 2.5)
			prompt = self.build_prompt_from_tags(image_tags, rng, n_tags)
		else:
			prompt = self.build_prompt_from_caption(image_caption, rng)
		
		# Add score tags
		# Dropped 10% of the time
		if rng.random() < 0.9:
			prompt = self.add_score_tags(image_score, prompt, rng)
		
		# UCG
		if random.random() < 0.05:
			prompt = ""
		
		# Tokenize the prompt
		tokens = self.tokenizer.encode(prompt, padding=False, truncation=False, add_special_tokens=False, verbose=False)
		tokens_2 = self.tokenizer_2.encode(prompt, padding=False, truncation=False, add_special_tokens=False, verbose=False)

		return {
			'latent': latent,
			'prompt': tokens,
			'prompt_2': tokens_2,
			'original_size': torch.tensor([original_height, original_width], dtype=torch.long),
			'crop': torch.tensor([crop_y, crop_x], dtype=torch.long),
			'target_size': torch.tensor([latent_width * 8, latent_height * 8], dtype=torch.long),   # goofed on height vs width; fixed by reversing here
		}
	
	def build_prompt_from_caption(self, caption: str, rng: random.Random) -> str:
		return caption
	
	def build_prompt_from_tags(self, tag_string: str, rng: random.Random, n_tags: int) -> str:
		# Prompt length tends to follow a normal distribution based on my measurements
		#n_tags = int(random.gauss(self.n_tags_mean, self.n_tags_std))
		n_tags = max(5, n_tags)  # Minimum of 5 tags

		# Split tag string into tags
		# Tags are shuffled, important tags are always included, and the number of tags is limited by n_tags
		tags = set(tag.strip() for tag in tag_string.split(",") if tag.strip())
		important_tags = tags.intersection(IMPORTANT_TAGS)
		n_tags = min(max(n_tags, len(important_tags)), len(tags))
		tags = list(important_tags) + random.sample(list(tags - important_tags), n_tags - len(important_tags))
		assert len(tags) <= n_tags, f"Expected {n_tags} tags, got {len(tags)}"
		random.shuffle(tags)

		# Prompt construction
		tag_type = rng.randint(0, 2)   # Use underscores, spaces, or mixed

		prompt = ""
		for tag in tags:
			# Randomly mutate tags using aliases
			if tag in self.tag_aliases and random.random() < 0.2:
				#old_tag = tag
				tag = random.choice(self.tag_aliases[tag])
				#print(f"Mutated tag {old_tag} to {tag}")
			
			# Regularize across tags with spaces or underscores, or mixed.
			if tag_type == 1:
				tag = tag.replace("_", " ")
			elif tag_type == 2:
				if random.random() < 0.8:
					tag = tag.replace("_", " ")
			
			if len(prompt) > 0:
				prompt += ","
				# Space between most times
				# NOTE: I don't think this matters because CLIP tokenizer ignores spaces?
				if random.random() < 0.8:
					prompt += ' '
				prompt += tag
			else:
				prompt += tag
				
		return prompt
	
	def add_score_tags(self, score: int, prompt: str, rng: random.Random) -> str:
		# Add score tag(s) to the front
		# E.g. score_9, score_9_up, score_8_up, etc.
		# score_N_up tags are inclusive (a score 9 image is score_9_up, score_8_up, etc.)
		# A random number of score tags are added, to regularize the model against overfitting to specific score tags sequences.
		# The end-user is likely to use a single tag, like score_9, or score_8_up, but we randomly include more than one
		# to hopefully help the model learn their meaning faster.
		score_tags = [f"score_{s}_up" for s in range(1, score+1)]
		score_tags.append(f"score_{score}")
		n_score_tags = random.choices([1, 2, 3], [3, 2, 1])[0]
		tags = random.sample(score_tags, min(n_score_tags, len(score_tags)))

		for tag in tags:
			# Regularize across score tags with spaces or underscores.
			if rng.random() < 0.2:
				tag = tag.replace("_", " ")
			
			# Regularize across score tags separated by commas or spaces.
			prompt = rng.choice([tag + ", ", tag + " ", tag + ","]) + prompt
		
		return prompt.strip()

	def collate_fn(self, batch: list[dict]) -> dict:
		latents = torch.stack([item['latent'] for item in batch])
		original_sizes = torch.stack([item['original_size'] for item in batch])
		crops = torch.stack([item['crop'] for item in batch])
		target_sizes = torch.stack([item['target_size'] for item in batch])

		# Target length for the prompts is based on the longest in the batch
		# Padded out to a multiple of 75
		target_length = max([len(x['prompt']) for x in batch])
		target_length += (75 - target_length % 75) if target_length % 75 != 0 else 0
		if target_length == 0:
			target_length = 75

		# Chunk up the prompts
		chunks = [chunk_tokens(item['prompt'], self.tokenizer, target_length) for item in batch]
		chunks_2 = [chunk_tokens(item['prompt_2'], self.tokenizer_2, target_length) for item in batch]

		# Stack the chunks
		chunks = torch.stack(chunks)
		chunks_2 = torch.stack(chunks_2)
		assert chunks.shape == (len(batch), target_length // 75, 77)

		# Truncate to a maximum of 3 chunks. In practice, prompts longer than 3 chunks are rare, and that's the limit that NovelAI uses.
		chunks = chunks[:, :3]
		chunks_2 = chunks_2[:, :3]
		assert chunks.shape == (len(batch), min(3, target_length // 75), 77)
		assert chunks_2.shape == chunks.shape

		return {
			'latent': latents,
			'prompt': chunks,
			'prompt_2': chunks_2,
			'original_size': original_sizes,
			'crop': crops,
			'target_size': target_sizes,
		}


def chunk_tokens(tokens: list[int], tokenizer: CLIPTokenizer, target_length: int) -> torch.Tensor:
	assert target_length % 75 == 0, "Target length must be a multiple of 75"
	chunks = []

	# Split into chunks of 75 tokens
	# Each of those chunks is bookended by BOS and EOS tokens
	# If any chunks are shorter than 77 tokens, pad them with pad tokens
	for i in range(0, target_length, 75):
		chunk = tokens[i:i+75]
		chunks.append(tokenizer.bos_token_id)
		chunks.extend(chunk)
		chunks.append(tokenizer.eos_token_id)
		chunks.extend([tokenizer.pad_token_id] * (75 - len(chunk)))
	
	#print(f"chunk_tokens: target_length: {target_length}, len(chunks): {len(chunks)}")
	
	# Convert to tensor (Nx77)
	tensor = torch.tensor(chunks, dtype=torch.long).view(-1, 77)
	assert tensor.shape == (target_length // 75, 77)

	return tensor


def read_tag_aliases() -> dict[str, str]:
	"""
	Returns a mapping based on tag aliases.
	This maps from aliased tags back to a canonical tag.
	Given a tag like "ff7" as key, for example, the value would be "final_fantasy_vii".
	"""
	aliases = [json.loads(line) for line in open('tag_aliases000000000000.json', 'r')]
	alias_map = {}

	for alias in aliases:
		if alias['status'] != 'active':
			continue

		assert alias['antecedent_name'] != alias['consequent_name'], "Self-aliases found in tag aliases"

		# Duplicate antecedent->consequent mappings are allowed, but only if they are the same
		# This is because the dataset contains a few duplicates (unknown why)
		assert alias['antecedent_name'] not in alias_map or alias_map[alias['antecedent_name']] == alias['consequent_name'], "Duplicate antecedents found in tag aliases"

		alias_map[alias['antecedent_name']] = alias['consequent_name']

	# Check for chains by ensuring that consequents are not also antecedents
	assert all(consequent not in alias_map for consequent in alias_map.values()), "Chains found in tag aliases"
	
	return alias_map


def gen_buckets(data) -> list[Bucket]:
	"""
	Organizes the dataset into buckets based on the latent size and caption length (approx. number of token chunks).
	They're bucketed by number of chunks, because a) it's slightly more efficinet, and b) it helps balance the distribution of chunk counts.
	If they weren't bucketed by chunk, the vast majority of batches would just pad the chunks to 2 or 3, underrepresenting chunk count 1.
	"""
	# NOTE: It's _much_ faster to extract the columns we need like this and zip them, than to naively iterate over the dataset
	widths = data['latent_width']
	heights = data['latent_height']
	n_tokens = data['n_tokens']

	buckets: dict[tuple[int, int, int], Bucket] = {}

	for i, (width, height, n_tokens) in tqdm(enumerate(zip(widths, heights, n_tokens))):
		# We assume ~5 tokens will be added by the score tags
		# It's obviously random, but we don't need to be super precise here; just close enough to get the batches lining up most of the time
		n_tokens = n_tokens + 5

		aspect = width / height
		resolution = (width, height)

		n_chunks = (n_tokens + 74) // 75
		n_chunks = min(max(n_chunks, 1), 3)   # Clamp to 1-3 chunks

		k = (width, height, n_chunks)

		if k not in buckets:
			buckets[k] = Bucket(resolution=resolution, aspect=aspect, n_chunks=n_chunks, images=[])
		
		buckets[k].images.append(i)
	
	return list(buckets.values())


class AspectBucketSampler(Sampler[list[tuple[tuple[int, int], int, int]]]):
	"""
	Samples batches from a dataset that has been split into aspect ratio buckets.
	Each batch will contain batch_size images from a single bucket (or less if ragged_batches is True)
	Images are shuffled within each bucket, if shuffle is True.
	The indices in each batch are tuples of (resolution, index, epoch), where resolution is the resolution of the bucket and index is the index of the image within the dataset.
	When the dataset uses randomization, the epoch is meant to be used to deterministically generate the randomization.
	When ragged_batches is False, epochs may have "leftover" images from various buckets that didn't fit into a batch. These images will be dropped for that epoch.

	Args:
		dataset: The dataset to sample from.
		buckets: The list of buckets.
		batch_size: The number of images per batch.
		num_replicas: The number of processes participating in distributed training.
		rank: The rank of the current process.
		shuffle: Whether to shuffle the images within each bucket.
		seed: The random seed to use for shuffling.
		ragged_batches: Whether to allow batches to be smaller than batch_size. If True, some batches may be smaller AND some replicas may receive fewer batches. If False, all batches will be the same size AND all replicas will receive the same number of batches, by dropping images when necessary.
	"""
	def __init__(
		self,
		dataset: Dataset,
		buckets: list[Bucket],
		batch_size: int,
		num_replicas: Optional[int] = None,
		rank: Optional[int] = None,
		shuffle: bool = True,
		seed: int = 0,
		ragged_batches: bool = False,
	) -> None:
		if num_replicas is None:
			if not dist.is_available():
				raise RuntimeError("Requires distributed package to be available")
			num_replicas = dist.get_world_size()
        
		if rank is None:
			if not dist.is_available():
				raise RuntimeError("Requires distributed package to be available")
			rank = dist.get_rank()
		
		if rank >= num_replicas or rank < 0:
			raise ValueError(f"Invalid rank {rank} for num_replicas {num_replicas}")
		
		self.dataset = dataset
		self.buckets = buckets
		self.num_replicas = num_replicas
		self.rank = rank
		self.epoch = 0
		self.batch_size = batch_size
		self.ragged_batches = ragged_batches
		self.shuffle = shuffle
		self.seed = seed
		self.resume_index = None

		if self.ragged_batches:
			total_batches = sum(int(math.ceil(len(bucket.images) / batch_size)) for bucket in buckets)
			self.num_samples = len(range(self.rank, total_batches, self.num_replicas))
		else:
			total_batches = sum(len(bucket.images) // batch_size for bucket in buckets)
			self.num_samples = total_batches // self.num_replicas
	
	def set_state(self, epoch: int, index: int) -> None:
		"""
		Sets the epoch and fast forwards the iterator to the given index.
		Needs to be called before the dataloader is iterated over.
		"""
		assert not self.ragged_batches, "set_state is not supported when ragged_batches is True, as it's a footgun"
		self.set_epoch(epoch)
		self.resume_index = index

	def __iter__(self) -> Iterator[list[tuple[tuple[int, int], int, int]]]:
		rng = random.Random(hash((self.seed, self.epoch))) if self.shuffle else None

		if rng is not None:
			# Make a copy of the buckets so we don't modify the original
			epoch_buckets = [Bucket(bucket.resolution, bucket.aspect, bucket.n_chunks, bucket.images[:]) for bucket in self.buckets]

			# Shuffle each bucket
			for bucket in epoch_buckets:
				rng.shuffle(bucket.images)
		else:
			epoch_buckets = self.buckets
		
		# Split all the buckets into batches
		batches = []
		leftovers = []

		for bucket in epoch_buckets:
			for i in range(0, len(bucket.images), self.batch_size):
				batch = bucket.images[i:i+self.batch_size]
				if len(batch) != self.batch_size and not self.ragged_batches:
					leftovers.extend(batch)
				else:
					batches.append((bucket.resolution, batch))
		
		# Shuffle the batches
		if rng is not None:
			rng.shuffle(batches)

		# Split the batches into chunks for each replica
		subset = batches[self.rank:len(batches):self.num_replicas]

		# At this point, batch_indices itself might be ragged if the number of batches isn't evenly divisible by the number of replicas
		# If we're not using ragged batches, we need to trim it down so all replicas have the same number of batches
		if not self.ragged_batches:
			chunk_size = len(batches) // self.num_replicas
			subset = subset[:chunk_size]
		
		# Handle resume logic
		if self.resume_index is not None:
			subset = subset[self.resume_index:]
			self.resume_index = None
		
		# Convert subset from a list of Tuple[Tuple[int, int], List[int]] to a list of List[Tuple[Tuple[int, int], int, int]]
		subset = [[(res, i, self.epoch) for i in batch] for res, batch in subset]
		
		return iter(subset)
	
	def __len__(self) -> int:
		return self.num_samples
	
	def set_epoch(self, epoch: int) -> None:
		self.epoch = epoch