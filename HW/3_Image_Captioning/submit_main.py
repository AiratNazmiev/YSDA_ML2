# Описание структуры присылаемого архива - для самопроверки
#  - submit_main.py
#  - vocab.tsv
#  - checkpoint

# ВАЖНО: если в любой функции есть параметры - не меняйте их порядок и не переименовывайте,
#   если требуется добавить ещё параметры, то добавляйте в конец и обязательно с установленными default-ами

# 0. Все необходимые import-ы
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import cv2
import os
import pandas as pd
import numpy as np
import random
import torchvision
from tqdm.auto import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr

import pandas as pd
import re
import os
import pickle

import warnings
warnings.filterwarnings("ignore")

# 1. Подготовка данных

## Как прочитать словарь, переданный вами внутри архива - используйте эту функцию в своём датасете
def tokenize(text: str) -> str:
    text1 = re.sub(r'[^\w\s]', ' ', text.lower())
    text2 = re.sub(r'\s+', ' ', text1).strip()
    words = re.findall(r'\b\w+\b', text2)
    
    return ['<BOS>', *words, '<EOS>']

def get_vocab(unzip_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    
    with open(os.path.join(unzip_root, "vocab.tsv"), 'rb') as file:
        tok_to_ind, ind_to_tok = pickle.load(file)
            
    return tok_to_ind, ind_to_tok

tok_to_ind, ind_to_tok = get_vocab('./')

## Ваш датасет
class ImageCaptioningDataset(Dataset):
    """
        imgs_path ~ путь к папке с изображениями
        captions_path ~ путь к .tsv файлу с заголовками изображений
    """
    def __init__(
        self, 
        imgs_path: str, 
        captions_path: str, 
        train: bool = True
    ) -> None:
        super(ImageCaptioningDataset).__init__()
        # Читаем и записываем из файлов в память класса, чтобы быстро обращаться внутри датасета
        # Если не хватает памяти на хранение всех изображений, то подгружайте прямо во время __getitem__, но это замедлит обучение
        # Проведите всю предобработку, которую можно провести без потери вариативности датасета, здесь
        
        self.train = train
        
        image_names = os.listdir(imgs_path)
        self.num_images = len(image_names)
        
        self.images = []
        
        self.image_prepare = tr.Compose([
            tr.ToPILImage(),
            tr.ToTensor(),
            tr.Resize(size=(232, 232), interpolation=tr.InterpolationMode.BILINEAR),
        ])
        
        channel_mean = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        
        self.image_train_augmentations = tr.Compose([
            tr.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.025, hue=0.02),
            tr.RandomRotation(degrees=15, interpolation=tr.InterpolationMode.BILINEAR),
            tr.Resize(size=(224, 224), antialias=True),
            tr.Normalize(mean=channel_mean, std=channel_std)
        ])
        
        self.image_prepare_val = tr.Compose([
            tr.ToPILImage(),
            tr.ToTensor(),
            tr.Resize(size=(224, 224), interpolation=tr.InterpolationMode.BILINEAR, antialias=True),
            tr.Normalize(mean=channel_mean, std=channel_std),
        ])
        
        self.image_val_augmentations = tr.Compose([
    
        ])
                
        for img_name in image_names:
            self.images.append(
                self.image_prepare(cv2.imread(os.path.join(imgs_path, img_name)))
            )
   
        captions_df = pd.read_csv(captions_path, sep='\t')
        
        if len(captions_df) != self.num_images:
            raise ValueError("Different number of images and captions")
        
        self.captions = []
        
        tok_to_ind, ind_to_tok = get_vocab('./')
        
        def to_ids(text: str) -> list[int]:
            tokens = tokenize(text)
            ids = []
            unk_idx = tok_to_ind['<UNK>']
            for t in tokens:
                idx = tok_to_ind.get(t)
                ids.append(idx if idx is not None else unk_idx)
            
            return ids
                    
        for idx in range(self.num_images):
            self.captions.append([])
            for n in range(1, captions_df.shape[1]):
                self.captions[idx].append(to_ids(captions_df.iloc[idx, n]))

    def __getitem__(self, index):
        # Получаем предобработанное изображение (не забудьте отличие при train=True или train=False)
        # Берём все заголовки или только один случайный (случайность должна происходить при каждом вызове __getitem__, 
        #  чтобы во время обучения вы в разных эпохах могли видеть разные заголовки для одного изображения)
        
        img_wo_aug = self.images[index]
        
        if self.train:
            img = self.image_train_augmentations(img_wo_aug)
        else:
            img = self.image_val_augmentations(img_wo_aug)
            
        #caption = random.choice(self.captions[index])
        caption = self.captions[index]
        
        return img, caption
    
    def __len__(self):
        return self.num_images

# Здесь хотим задать кастомную функцию для того, как именно складывать данные в батч
# Эта функция позже будет передана в collate_fn аргумент даталоадера и будет отвечать за то,
#  как обработать батч и превратить его в тензоры нужного вида

def collate_fn(batch):
    # Функция получает на вход batch - представляет из себя List[el], где каждый el - один вызов __getitem__
    #  вашего датасета
    # На выход вы выдаёте то, что будет выдавать Dataloader на каждом next() из генератора - вы хотите иметь на выходе
    #  несколько тензоров
    
    # Моё предложение по тому как должен выглядеть батч на выходе:
    #   img_batch: [batch_size, num_channels, height, width] --> сложенные в батч изображения
    #   captions_batch: [batch_size, num_captions_per_image, max_seq_len or local_max_seq_len] --> сложенные в
    #       батч заголовки при помощи padding-а
    
    
    images, captions = zip(*batch)
    
    img_batch = torch.stack(images, dim=0)
    captions_batch = nn.utils.rnn.pad_sequence([torch.tensor(c) for c in captions], batch_first=True, padding_value=tok_to_ind['<PAD>'])
    # it's better to work with packed sequences
    
    return img_batch, captions_batch[:, None, :]

def get_val_dataloader(dataset, batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

# 2. Построение модели

## Аргументы для общего класса
init_kwargs = dict()

## Общий класс модели

IMG_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 300
VOCAB_SIZE = 3478

class img_fe_class(nn.Module):
    def __init__(
        self, 
        text_embedding_dim: int = IMG_EMBEDDING_DIM,
        head_hidden_size: int = 1024
    ) -> None:
        super().__init__()
        
        mobilenet_v3_large = torchvision.models.mobilenet_v3_large(
            #weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT,
            #progress=False
        )
        
        self.backbone = nn.Sequential(
            mobilenet_v3_large.features,
            mobilenet_v3_large.avgpool,
            nn.Flatten(start_dim=1, end_dim=3)
        )
        backbone_out_features = mobilenet_v3_large.classifier[0].in_features  # 960
        
        for p in self.backbone[0][:-1].parameters():
            p.requires_grad = False
                        
        self.head = nn.Sequential(
            nn.Linear(
                in_features=backbone_out_features,
                out_features=head_hidden_size
            ),
            nn.BatchNorm1d(num_features=head_hidden_size),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=head_hidden_size, out_features=text_embedding_dim)
        )
        
    def forward(self, imgs):
        return self.head(self.backbone(imgs))
    
    
class text_fe_class(nn.Module):
    def __init__(
        self,
        text_embedding_dim: int = TEXT_EMBEDDING_DIM,
        img_embedding_dim: int = IMG_EMBEDDING_DIM,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = 1024,
        num_layers: int = 2,
        bias: int = True,
        batch_first: bool = True,
        dropout: float = 0.5,
        img_embedding_hidden_size: int = 1024,
        #head_hidden_size: int = 1024,
        max_seq_length: int = 30
    ) -> None:
        super().__init__()
        
        self.max_seq_length = max_seq_length  # ограничим число токенов в выходе при генерации
        
        self.text_embed = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=text_embedding_dim, 
            padding_idx=tok_to_ind['<PAD>']
        )
        
        self.text_embed.weight = nn.Parameter(
            torch.randn((VOCAB_SIZE, TEXT_EMBEDDING_DIM)),
            #torch.from_numpy(glove_weights).to(dtype=self.text_embed.weight.dtype),
            requires_grad=True,
        )
        
        self.img_embed_fc = nn.Sequential(
            nn.Linear(
                in_features=img_embedding_dim,
                out_features=img_embedding_hidden_size
            ),
            nn.BatchNorm1d(num_features=img_embedding_hidden_size),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=img_embedding_hidden_size,
                out_features=num_layers * hidden_size  
            )
        )
        
        self.h_0_reshape = num_layers, hidden_size

        self.gru = nn.GRU(
            input_size=text_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.,
            bidirectional=False
        )
        
        
        # self.head = nn.Sequential(
        #     nn.Linear(
        #         in_features=hidden_size,
        #         out_features=head_hidden_size
        #     ),
        #     nn.BatchNorm1d(normalized_shape=head_hidden_size),
        #     nn.Hardswish(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(
        #         in_features=head_hidden_size,
        #         out_features=vocab_size
        #     )
        # )
        self.head = nn.Sequential(
            nn.Linear(
                in_features=hidden_size,
                out_features=vocab_size
            )
        )
        
    def _get_h0_img(self, img_features):
        h_0_img = self.img_embed_fc(img_features)
        h_0_img = h_0_img.reshape(-1, *self.h_0_reshape).permute(1, 0, 2).contiguous()
        
        return h_0_img
        
    def forward(self, texts, img_features):
        texts_emb = self.text_embed(texts)
        
        h_0_img = self._get_h0_img(img_features)
        
        gru_output, _ = self.gru(texts_emb.squeeze(1), h_0_img)
        
        logits = self.head(gru_output)
        
        return logits[:, None, :]


class image_captioning_model(nn.Module):
    def __init__(
        self,
            text_embedding_dim: int = IMG_EMBEDDING_DIM,
            img_embedding_dim: int = IMG_EMBEDDING_DIM,
            vocab_size: int = VOCAB_SIZE,
            img_fe_head_hidden_size: int = 256,
            text_fe_hidden_size: int = 256,
            text_fe_num_layers: int = 2,
            text_fe_bias: int = True,
            text_fe_batch_first: bool = True,
            text_fe_dropout: float = 0.3,
            text_fe_img_embedding_hidden_size: int = 512,
            text_fe_head_hidden_size: int = 1024,
            max_seq_length: int = 30
        ) -> None:
        super().__init__()
        
        self.img_fe = img_fe_class(
            # text_embedding_dim=text_embedding_dim,
            # head_hidden_size=img_fe_head_hidden_size
        )
        
        self.text_fe = text_fe_class(
            # text_embedding_dim=text_embedding_dim,
            # img_embedding_dim=img_embedding_dim,
            # vocab_size=vocab_size,
            # hidden_size=text_fe_hidden_size,
            # num_layers=text_fe_num_layers,
            # bias=text_fe_bias,
            # batch_first=text_fe_batch_first,
            # dropout=text_fe_dropout,
            # img_embedding_hidden_size=text_fe_img_embedding_hidden_size,
            # head_hidden_size=text_fe_head_hidden_size,
            # max_seq_length=max_seq_length
        )
        
    def encode(self, img_batch: torch.Tensor) -> torch.Tensor:
        return self.img_fe(img_batch)
    
    def decode(self, img_features: torch.Tensor, texts_batch: torch.Tensor) -> torch.Tensor:
        return self.text_fe(texts_batch, img_features)
        
    def forward(self, img_batch: torch.Tensor, texts_batch: torch.Tensor) -> torch.Tensor:
        img_features = self.encode(img_batch)
        text_logits = self.decode(img_features, texts_batch)
        
        return text_logits

# # 3. Обучение модели

## Сборка вашей модели с нужными параметрами и подгрукой весов из чекпоинта
def get_model(unzip_root: str):
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    model = image_captioning_model()
    with open(os.path.join(unzip_root, "model.pth"), 'rb') as file:
        model.load_state_dict(torch.load(file, weights_only=False, map_location="cpu")['model_state_dict'])
    #model.load_state_dict(torch.load(os.path.join(unzip_root, "model.pth"), weights_only=False)['model_state_dict'])
        
    return model

# # 4. Оценка результатов

def tokens_to_text(ids: list[int]) -> str:
    text = []
    for idx in ids:
        text.append(ind_to_tok[idx])
        
    return ' '.join(text)

image_prepare_val = tr.Compose([
    tr.ToPILImage(),
    tr.ToTensor(),
    tr.Resize(size=(224, 224), interpolation=tr.InterpolationMode.BILINEAR),
    tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## Генерация предсказания по картинке
from typing import Optional

@torch.no_grad()
def generate(
    model: nn.Module,
    image: np.ndarray,
    max_seq_len: Optional[int] = 15,#30,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> tuple[torch.Tensor, str]:
    texts = []
    tokens = []
    for _ in range(5):
        tkn, txt = generate_(model, image, max_seq_len, top_p, top_k)
        texts.append(txt)
        tokens.append(tkn[0, 0].tolist())
        
    return tokens, texts


@torch.no_grad()
def generate_(
    model: nn.Module,
    image: np.ndarray,
    max_seq_len: Optional[int] = 15,#30,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> tuple[torch.Tensor, str]:
    """
    По картинке image генерируете текст моделью model либо пока не сгенерируете '<EOS>' токен, либо пока не сгенерируете max_seq_len токенов
        top_k -> после получения предсказания оставляете первые top_k слов и сэмплируете случайно с перенормированными вероятностями из оставшихся слов
        top_p -> после получения предсказания оставляете первые сколько-то слов, так, чтобы суммарная вероятность оставшихся слов была не больше top_p,
            после чего сэмплируете с перенормированными вероятностями из оставшихся слов
        иначе -> сэмплируете случайное слово с предсказанными вероятностями
    """
    assert top_p is None or top_k is None, "Don't use top_p and top_k at the same time"
    
    if max_seq_len is None:
        max_seq_len = 1024  # Arbitrary large number if not specified
    
    model.eval()
    
    image_transformed = image_prepare_val(image).unsqueeze(0)
    
    img_features = model.encode(image_transformed)
    
    generated = [tok_to_ind['<BOS>']]
    eos_tok_idx = tok_to_ind['<EOS>']
    
    for _ in range(max_seq_len):
        text_tokens = torch.tensor(generated)[None, None, :]
        
        text_logits = model.decode(img_features, text_tokens)[0]
        
        next_token_logits = text_logits[0, -1, :]
        
        probs = torch.softmax(next_token_logits, dim=-1)
        
        if top_k is not None and top_k > 0:
            top_k_probs, top_k_ids = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs)
            probs[..., top_k_ids] = top_k_probs
            probs[top_k_ids] /= probs[top_k_ids].sum()
        
        elif top_p is not None and top_p > 0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            

            sorted_indices_to_remove = cumulative_probs > top_p

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs = probs.clone()
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()
            
        next_token = torch.multinomial(probs, num_samples=1).item()
            
        generated.append(next_token)
        if next_token == eos_tok_idx:
            break
        
    result_text = tokens_to_text(text_tokens[0, 0].cpu().numpy().tolist()[1:])
    
    return text_tokens, result_text

# @torch.no_grad()
# def generate(
#     model: nn.Module,
#     image: np.ndarray,
#     max_seq_len: Optional[int] = 30,
#     top_p: Optional[float] = None,
#     top_k: Optional[int] = None,
# ) -> tuple[torch.Tensor, str]:
#     """
#     По картинке image генерируете текст моделью model либо пока не сгенерируете '<EOS>' токен, либо пока не сгенерируете max_seq_len токенов
#         top_k -> после получения предсказания оставляете первые top_k слов и сэмплируете случайно с перенормированными вероятностями из оставшихся слов
#         top_p -> после получения предсказания оставляете первые сколько-то слов, так, чтобы суммарная вероятность оставшихся слов была не больше top_p,
#             после чего сэмплируете с перенормированными вероятностями из оставшихся слов
#         иначе -> сэмплируете случайное слово с предсказанными вероятностями
#     """
#     assert top_p is None or top_k is None, "Don't use top_p and top_k at the same time"
    
#     if max_seq_len is None:
#         max_seq_len = 1024  # Arbitrary large number if not specified
    
#     model.eval()
    
#     image_transformed = image_prepare_val(image).unsqueeze(0)
    
#     img_features = model.encode(image_transformed)
    
#     generated = [tok_to_ind['<BOS>']]
#     eos_tok_idx = tok_to_ind['<EOS>']
    
#     for _ in range(max_seq_len):
#         text_tokens = torch.tensor(generated)[None, None, :]
        
#         text_logits = model.decode(img_features, text_tokens)[0]
        
#         next_token_logits = text_logits[0, -1, :]
        
#         probs = torch.softmax(next_token_logits, dim=-1)
        
#         if top_k is not None and top_k > 0:
#             top_k_probs, top_k_ids = torch.topk(probs, top_k)
#             probs = torch.zeros_like(probs)
#             probs[..., top_k_ids] = top_k_probs
#             probs[top_k_ids] /= probs[top_k_ids].sum()
        
#         elif top_p is not None and top_p > 0:
#             sorted_probs, sorted_indices = torch.sort(probs, descending=True)
#             cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            

#             sorted_indices_to_remove = cumulative_probs > top_p

#             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#             sorted_indices_to_remove[..., 0] = False
            
#             indices_to_remove = sorted_indices[sorted_indices_to_remove]
#             probs = probs.clone()
#             probs[indices_to_remove] = 0
#             probs = probs / probs.sum()
            
#         next_token = torch.multinomial(probs, num_samples=1).item()
            
#         generated.append(next_token)
#         if next_token == eos_tok_idx:
#             break
        
#     result_text = tokens_to_text(text_tokens[0, 0].cpu().numpy().tolist()[1:])
    
#     return text_tokens, result_text

# @torch.no_grad()
# def generate_beam_search(
#     model: nn.Module,
#     image: np.ndarray,
#     max_seq_len: Optional[int] = 30,
#     beam_width: int = 10,
# ) -> tuple[torch.Tensor, str]:
#     model.eval()

#     image_transformed = image_prepare_val(image).unsqueeze(0)
#     img_features = model.encode(image_transformed)
    
#     bos_tok_idx = tok_to_ind['<BOS>']
#     eos_tok_idx = tok_to_ind['<EOS>']

#     beams = torch.full((beam_width, max_seq_len), fill_value=0, dtype=torch.int64)
#     beams[:, 0] = bos_tok_idx
#     beam_log_probs = torch.zeros(beam_width, dtype=torch.float64)
#     beam_completed = torch.full((beam_width, ), fill_value=False)

#     for idx in range(1, max_seq_len):
#         all_candidates = []
#         for beam_i in range(beam_width):
#             if beam_completed[beam_i]:
#                 all_candidates.append((beam_log_probs[beam_i], beams[beam_i].clone(), True))
#                 continue

#             current_seq = beams[beam_i, :idx].unsqueeze(0)  # [1, idx]

#             logits = model.decode(
#                 img_features, current_seq.unsqueeze(1)
#             ).squeeze(1)  # [1, seq_len, vocab_size]
            
#             next_token_logits = logits[:, -1, :]  # [1, vocab_size]
#             log_probs = torch.log_softmax(next_token_logits, dim=-1)  # [1, vocab_size]
            
#             topk_log_probs, topk_token_ids = torch.topk(log_probs, k=beam_width, dim=-1)
#             topk_log_probs = topk_log_probs.squeeze(0)  # [beam_width]
#             topk_token_ids = topk_token_ids.squeeze(0)  # [beam_width]

#             for i in range(beam_width):
#                 candidate_seq = beams[beam_i].clone()
#                 candidate_seq[idx] = topk_token_ids[i]
#                 candidate_score = beam_log_probs[beam_i] + topk_log_probs[i]
#                 candidate_complete = (topk_token_ids[i].item() == eos_tok_idx)
#                 all_candidates.append((candidate_score, candidate_seq, candidate_complete))
        
#         all_candidates.sort(key=lambda x: x[0], reverse=True)
#         selected = all_candidates[:beam_width]
#         beams = torch.stack([cand[1] for cand in selected])
#         beam_log_probs = torch.tensor([cand[0] for cand in selected], dtype=torch.float64)
#         beam_completed = torch.tensor([c[2] for c in selected])
        
#         if beam_completed.all():
#             break

#     best_beam_idx = torch.argmax(beam_log_probs)
#     best_beam = beams[best_beam_idx]

#     if eos_tok_idx in best_beam:
#         eos_idx = torch.argwhere(best_beam == eos_tok_idx)[0]
#         best_beam = best_beam[:eos_idx]
        
#     result_text = tokens_to_text(best_beam[1:].cpu().numpy().tolist())

#     return best_beam[None, None, :], result_text
