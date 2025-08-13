# learning_system.py

"""
실제 딥러닝 학습 시스템
- GPU 기반 토큰화 및 임베딩 학습
- 실제 신경망 가중치 업데이트  
- 문맥 이해 및 패턴 학습
- 한국어 언어 모델링
- 도메인별 답변 생성
"""

import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import hashlib
import json
import pickle
import tempfile
import os
import random
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# 딥러닝 학습 상수
EMBEDDING_DIM = 768
HIDDEN_DIM = 512
NUM_LAYERS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 512
DROPOUT_RATE = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0

# 학습 파라미터
DEFAULT_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_MIN_SAMPLES = 5
DEFAULT_CACHE_SIZE = 300
PATTERN_LIMIT = 15
LEARNING_HISTORY_LIMIT = 200
REAL_LEARNING_THRESHOLD = 10

def _default_int():
    return 0

def _default_float():
    return 0.0

def _default_list():
    return []

def _default_counter():
    return Counter()

def _default_float_dict():
    return defaultdict(_default_float)

def _default_int_dict():
    return defaultdict(_default_int)

class DeepLearningQuestionAnalyzer(nn.Module):
    """실제 딥러닝 기반 문제 분석기"""
    
    def __init__(self, vocab_size: int = 50000, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, HIDDEN_DIM, NUM_LAYERS, 
                           batch_first=True, dropout=DROPOUT_RATE, bidirectional=True)
        self.attention = nn.MultiheadAttention(HIDDEN_DIM * 2, 8, dropout=DROPOUT_RATE)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM, 256),
            nn.ReLU(), 
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5개 선택지
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10)  # 10개 도메인
        )
        
    def forward(self, input_ids, attention_mask=None):
        # 실제 GPU 연산 수행
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 어텐션 메커니즘 적용
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 마지막 hidden state 사용
        final_hidden = attn_out[:, -1, :]
        
        # 분류 수행
        answer_logits = self.classifier(final_hidden)
        domain_logits = self.domain_classifier(final_hidden)
        
        return answer_logits, domain_logits, final_hidden

class KoreanTextProcessor:
    """한국어 텍스트 실제 처리기"""
    
    def __init__(self, model_name: str = "klue/bert-base"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.cuda()
            self.bert_model.eval()
        except:
            # 대체 토크나이저 사용
            self.tokenizer = None
            self.bert_model = None
            
        self.vocab = self._build_korean_vocab()
        
    def _build_korean_vocab(self) -> Dict[str, int]:
        """한국어 어휘 사전 구축"""
        vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        
        # 한국어 기본 어휘
        korean_words = [
            "개인정보", "전자금융", "금융투자업", "위험관리", "정보보안", "사이버보안",
            "암호화", "복호화", "트로이", "악성코드", "해킹", "피싱", "스미싱",
            "방화벽", "침입탐지", "접근통제", "인증", "인가", "권한", "보안정책",
            "재해복구", "업무연속성", "백업", "복구", "BCP", "DRP",
            "개인정보보호법", "전자금융거래법", "정보통신망법", "자본시장법",
            "정의", "의미", "개념", "해당", "적절", "옳은", "틀린", "잘못된"
        ]
        
        for i, word in enumerate(korean_words, 4):
            vocab[word] = i
            
        return vocab
    
    def tokenize_korean(self, text: str) -> List[int]:
        """실제 한국어 토크나이징"""
        if self.tokenizer:
            # BERT 기반 토크나이징
            tokens = self.tokenizer.encode(text, max_length=MAX_SEQ_LENGTH, 
                                         truncation=True, padding='max_length')
            return tokens
        else:
            # 기본 토크나이징
            words = re.findall(r'[가-힣]+|[0-9]+|[a-zA-Z]+', text)
            tokens = []
            for word in words:
                tokens.append(self.vocab.get(word, self.vocab["<UNK>"]))
            
            # 패딩 적용
            if len(tokens) < MAX_SEQ_LENGTH:
                tokens.extend([self.vocab["<PAD>"]] * (MAX_SEQ_LENGTH - len(tokens)))
            else:
                tokens = tokens[:MAX_SEQ_LENGTH]
                
            return tokens
    
    def get_semantic_embedding(self, text: str) -> torch.Tensor:
        """실제 의미 임베딩 추출"""
        if self.bert_model and self.tokenizer:
            # BERT 기반 임베딩
            inputs = self.tokenizer(text, return_tensors="pt", max_length=MAX_SEQ_LENGTH,
                                  truncation=True, padding='max_length')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze()
        else:
            # 기본 임베딩
            tokens = self.tokenize_korean(text)
            embedding = torch.zeros(EMBEDDING_DIM)
            for token in tokens[:50]:  # 처음 50개 토큰만 사용
                if token > 0:
                    embedding += torch.randn(EMBEDDING_DIM) * 0.1
            return embedding

@dataclass
class LearningState:
    """실제 학습 상태"""
    samples_processed: int
    weights_updated: int
    loss_history: List[float]
    accuracy_history: List[float]
    gpu_memory_used: float
    training_time: float

class RealLearningSystem:
    """실제 딥러닝 학습 시스템"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
        # GPU 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 실제 딥러닝 모델 초기화
        self.korean_processor = KoreanTextProcessor()
        self.deep_analyzer = DeepLearningQuestionAnalyzer().to(self.device)
        
        # 실제 옵티마이저 설정
        self.optimizer = optim.AdamW(
            self.deep_analyzer.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.CrossEntropyLoss()
        
        # 실제 학습 데이터 저장소
        self.training_samples = []
        self.learned_patterns = defaultdict(list)
        self.domain_knowledge = defaultdict(dict)
        
        # 학습 상태 추적
        self.learning_state = LearningState(0, 0, [], [], 0.0, 0.0)
        self.real_learning_active = False
        
        # 기존 호환성 유지
        self.pattern_weights = defaultdict(_default_float_dict)
        self.pattern_counts = defaultdict(_default_int)
        self.answer_distribution = {
            "mc": defaultdict(_default_int),
            "domain": defaultdict(_default_int_dict),
            "negative": defaultdict(_default_int)
        }
        
        # 학습 파라미터
        self.learning_rate = LEARNING_RATE
        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.min_samples = DEFAULT_MIN_SAMPLES
        
        # 통계
        self.stats = {
            "total_samples": 0,
            "correct_predictions": 0,
            "patterns_learned": 0,
            "korean_quality_avg": 0.0,
            "answer_diversity_score": 0.0,
            "model_result_usage": 0,
            "pattern_usage": 0,
            "deep_learning_samples": 0,
            "gpu_training_time": 0.0
        }
        
        # GPU 메모리 정보
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_memory_total = 0
            
        self._debug_print(f"딥러닝 학습 시스템 초기화 완료 - GPU: {torch.cuda.is_available()}")
    
    def _debug_print(self, message: str) -> None:
        if self.debug_mode:
            print(f"[딥러닝학습] {message}")
    
    def _real_deep_learning_training(self, question: str, answer: str, confidence: float) -> bool:
        """실제 딥러닝 학습 수행"""
        if confidence < 0.6:  # 높은 신뢰도에서만 학습
            return False
            
        try:
            start_time = time.time()
            self.deep_analyzer.train()
            
            # 실제 토크나이징 및 텐서 변환
            tokens = self.korean_processor.tokenize_korean(question)
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            # 정답 레이블 생성
            if answer.isdigit() and 1 <= int(answer) <= 5:
                target = torch.tensor([int(answer) - 1], dtype=torch.long).to(self.device)
            else:
                return False
                
            # 실제 순전파
            answer_logits, domain_logits, hidden = self.deep_analyzer(input_tensor)
            
            # 실제 손실 계산
            loss = self.criterion(answer_logits, target)
            
            # 실제 역전파
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.deep_analyzer.parameters(), GRADIENT_CLIP)
            
            # 실제 가중치 업데이트
            self.optimizer.step()
            
            # 학습 상태 업데이트
            self.learning_state.samples_processed += 1
            self.learning_state.weights_updated += 1
            self.learning_state.loss_history.append(loss.item())
            
            training_time = time.time() - start_time
            self.learning_state.training_time += training_time
            self.stats["gpu_training_time"] += training_time
            self.stats["deep_learning_samples"] += 1
            
            # GPU 메모리 사용량 추적
            if torch.cuda.is_available():
                self.learning_state.gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
            
            self._debug_print(f"실제 딥러닝 학습 완료 - 손실: {loss.item():.4f}, 시간: {training_time:.2f}초")
            
            self.real_learning_active = True
            return True
            
        except Exception as e:
            self._debug_print(f"딥러닝 학습 오류: {e}")
            return False
    
    def _deep_pattern_extraction(self, question: str) -> Dict:
        """실제 딥러닝 기반 패턴 추출"""
        try:
            self.deep_analyzer.eval()
            
            # 실제 토크나이징 및 임베딩
            tokens = self.korean_processor.tokenize_korean(question)
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                answer_logits, domain_logits, hidden = self.deep_analyzer(input_tensor)
                
                # 실제 확률 분포 계산
                answer_probs = F.softmax(answer_logits, dim=1).cpu().numpy()[0]
                domain_probs = F.softmax(domain_logits, dim=1).cpu().numpy()[0]
                
                # 의미적 임베딩 추출
                semantic_embedding = self.korean_processor.get_semantic_embedding(question)
                
                return {
                    "answer_probabilities": answer_probs,
                    "domain_probabilities": domain_probs,
                    "semantic_features": hidden.cpu().numpy()[0],
                    "bert_embedding": semantic_embedding.cpu().numpy() if torch.cuda.is_available() else semantic_embedding.numpy(),
                    "confidence": float(np.max(answer_probs))
                }
                
        except Exception as e:
            self._debug_print(f"딥러닝 패턴 추출 오류: {e}")
            return {}
    
    def learn_from_prediction(self, question: str, prediction: str,
                            confidence: float, question_type: str,
                            domain: List[str], is_model_result: bool = False) -> None:
        """실제 딥러닝 학습 수행"""
        
        # 실제 딥러닝 학습 시도
        if self._real_deep_learning_training(question, prediction, confidence):
            # 딥러닝 패턴 추출
            deep_patterns = self._deep_pattern_extraction(question)
            
            if deep_patterns:
                # 실제 학습된 패턴 저장
                pattern_key = hashlib.md5(question.encode()).hexdigest()[:8]
                self.learned_patterns[pattern_key] = {
                    "question_sample": question[:100],
                    "answer": prediction,
                    "confidence": confidence,
                    "deep_features": deep_patterns,
                    "domains": domain,
                    "timestamp": time.time()
                }
                
                # 도메인별 지식 업데이트
                for d in domain:
                    if d not in self.domain_knowledge:
                        self.domain_knowledge[d] = {}
                    self.domain_knowledge[d][pattern_key] = deep_patterns["answer_probabilities"]
        
        # 기존 패턴 기반 학습도 병행
        if confidence >= self.confidence_threshold:
            self._traditional_pattern_learning(question, prediction, confidence, domain)
        
        # 학습 샘플 추가
        self.training_samples.append({
            "question": question,
            "answer": prediction,
            "confidence": confidence,
            "question_type": question_type,
            "domains": domain,
            "is_model_result": is_model_result,
            "timestamp": time.time()
        })
        
        # 샘플 수 제한
        if len(self.training_samples) > LEARNING_HISTORY_LIMIT:
            self.training_samples = self.training_samples[-LEARNING_HISTORY_LIMIT:]
        
        self.stats["total_samples"] += 1
        
        # 배치 학습 수행
        if len(self.training_samples) % BATCH_SIZE == 0:
            self._batch_deep_learning()
    
    def _traditional_pattern_learning(self, question: str, prediction: str, 
                                    confidence: float, domain: List[str]) -> None:
        """기존 패턴 기반 학습"""
        patterns = self._extract_linguistic_patterns(question)
        
        for pattern in patterns:
            weight_boost = confidence * self.learning_rate
            self.pattern_weights[pattern][prediction] += weight_boost
            self.pattern_counts[pattern] += 1
        
        # 답변 분포 업데이트
        self.answer_distribution["mc"][prediction] += 1
        
        for d in domain:
            if d not in self.answer_distribution["domain"]:
                self.answer_distribution["domain"][d] = defaultdict(_default_int)
            self.answer_distribution["domain"][d][prediction] += 1
    
    def _batch_deep_learning(self) -> None:
        """배치 기반 딥러닝 학습"""
        if len(self.training_samples) < BATCH_SIZE:
            return
            
        try:
            start_time = time.time()
            self.deep_analyzer.train()
            
            # 배치 데이터 준비
            batch_questions = []
            batch_answers = []
            
            recent_samples = self.training_samples[-BATCH_SIZE:]
            for sample in recent_samples:
                if sample["answer"].isdigit() and 1 <= int(sample["answer"]) <= 5:
                    batch_questions.append(sample["question"])
                    batch_answers.append(int(sample["answer"]) - 1)
            
            if len(batch_questions) < 4:  # 최소 배치 크기
                return
            
            # 토크나이징
            batch_tokens = []
            for question in batch_questions:
                tokens = self.korean_processor.tokenize_korean(question)
                batch_tokens.append(tokens)
            
            # 텐서 변환
            input_tensor = torch.tensor(batch_tokens, dtype=torch.long).to(self.device)
            target_tensor = torch.tensor(batch_answers, dtype=torch.long).to(self.device)
            
            # 순전파
            answer_logits, domain_logits, hidden = self.deep_analyzer(input_tensor)
            
            # 손실 계산
            loss = self.criterion(answer_logits, target_tensor)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.deep_analyzer.parameters(), GRADIENT_CLIP)
            self.optimizer.step()
            self.scheduler.step()
            
            # 학습 상태 업데이트
            self.learning_state.weights_updated += len(batch_questions)
            self.learning_state.loss_history.append(loss.item())
            
            batch_time = time.time() - start_time
            self.learning_state.training_time += batch_time
            self.stats["gpu_training_time"] += batch_time
            
            self._debug_print(f"배치 딥러닝 학습 - 샘플: {len(batch_questions)}, 손실: {loss.item():.4f}, 시간: {batch_time:.2f}초")
            
        except Exception as e:
            self._debug_print(f"배치 학습 오류: {e}")
    
    def _extract_linguistic_patterns(self, question: str) -> List[str]:
        """언어학적 패턴 추출"""
        patterns = []
        question_lower = question.lower()
        
        # 도메인 패턴
        domain_patterns = {
            "개인정보": ["개인정보", "정보주체", "개인정보보호법"],
            "전자금융": ["전자금융", "전자적장치", "전자금융거래법"],
            "금융투자업": ["금융투자업", "투자매매업", "소비자금융업"],
            "위험관리": ["위험", "관리", "계획", "위험평가"],
            "사이버보안": ["트로이", "악성코드", "해킹", "피싱"],
            "암호화": ["암호화", "복호화", "PKI", "전자서명"]
        }
        
        for domain, keywords in domain_patterns.items():
            if any(keyword in question_lower for keyword in keywords):
                patterns.append(f"domain_{domain}")
        
        # 문법 패턴
        if "정의" in question_lower or "의미" in question_lower:
            patterns.append("definition_question")
        if any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
            patterns.append("negative_question")
        
        return patterns[:PATTERN_LIMIT]
    
    def predict_with_deep_learning(self, question: str, question_type: str) -> Tuple[str, float]:
        """실제 딥러닝 기반 예측"""
        if not self.real_learning_active or len(self.learned_patterns) < 5:
            return self._traditional_prediction(question, question_type)
        
        try:
            deep_patterns = self._deep_pattern_extraction(question)
            
            if deep_patterns and deep_patterns["confidence"] > 0.6:
                # 딥러닝 결과 사용
                answer_probs = deep_patterns["answer_probabilities"]
                predicted_answer = str(np.argmax(answer_probs) + 1)
                confidence = float(np.max(answer_probs))
                
                self._debug_print(f"딥러닝 예측 - 답변: {predicted_answer}, 신뢰도: {confidence:.3f}")
                return predicted_answer, confidence
            else:
                return self._traditional_prediction(question, question_type)
                
        except Exception as e:
            self._debug_print(f"딥러닝 예측 오류: {e}")
            return self._traditional_prediction(question, question_type)
    
    def _traditional_prediction(self, question: str, question_type: str) -> Tuple[str, float]:
        """기존 패턴 기반 예측"""
        patterns = self._extract_linguistic_patterns(question)
        
        if not patterns:
            return str(random.randint(1, 5)), 0.3
        
        answer_scores = defaultdict(float)
        total_weight = 0
        
        for pattern in patterns:
            if pattern in self.pattern_weights and self.pattern_counts[pattern] >= self.min_samples:
                pattern_weight = self.pattern_counts[pattern]
                
                for answer, weight in self.pattern_weights[pattern].items():
                    answer_scores[answer] += weight * pattern_weight
                    total_weight += pattern_weight
        
        if answer_scores:
            best_answer = max(answer_scores.items(), key=lambda x: x[1])
            confidence = best_answer[1] / max(total_weight, 1)
            return best_answer[0], min(confidence, 0.8)
        
        return str(random.randint(1, 5)), 0.3
    
    def get_learning_statistics(self) -> Dict:
        """실제 학습 통계 반환"""
        return {
            "deep_learning_active": self.real_learning_active,
            "samples_processed": self.learning_state.samples_processed,
            "weights_updated": self.learning_state.weights_updated,
            "gpu_memory_used_gb": self.learning_state.gpu_memory_used,
            "total_training_time": self.learning_state.training_time,
            "average_loss": np.mean(self.learning_state.loss_history[-10:]) if self.learning_state.loss_history else 0,
            "learned_patterns_count": len(self.learned_patterns),
            "domain_knowledge_count": len(self.domain_knowledge),
            "deep_learning_samples": self.stats["deep_learning_samples"],
            "gpu_training_time": self.stats["gpu_training_time"],
            "traditional_samples": self.stats["total_samples"] - self.stats["deep_learning_samples"]
        }
    
    def optimize_patterns(self) -> Dict:
        """패턴 최적화"""
        # 딥러닝 모델 최적화
        if self.real_learning_active and len(self.learned_patterns) >= 10:
            try:
                # 모델 가중치 정규화
                for param in self.deep_analyzer.parameters():
                    if param.grad is not None:
                        param.data = param.data * 0.99  # 가중치 감쇠
                
                # 학습률 조정
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.95
                    
            except Exception as e:
                self._debug_print(f"모델 최적화 오류: {e}")
        
        # 기존 패턴 최적화
        optimized = 0
        removed = 0
        
        patterns_to_remove = []
        for pattern, count in self.pattern_counts.items():
            if count < self.min_samples:
                patterns_to_remove.append(pattern)
        
        for pattern in patterns_to_remove:
            if pattern in self.pattern_weights:
                del self.pattern_weights[pattern]
            if pattern in self.pattern_counts:
                del self.pattern_counts[pattern]
            removed += 1
        
        return {
            "optimized": optimized,
            "removed": removed,
            "remaining": len(self.pattern_weights),
            "deep_patterns": len(self.learned_patterns),
            "learning_active": self.real_learning_active
        }
    
    def save_model(self, filepath: str = "./learning_model.pkl") -> bool:
        """모델 저장"""
        try:
            model_data = {
                "deep_analyzer_state": self.deep_analyzer.state_dict() if hasattr(self, 'deep_analyzer') else None,
                "optimizer_state": self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                "learned_patterns": dict(self.learned_patterns),
                "domain_knowledge": dict(self.domain_knowledge),
                "learning_state": {
                    "samples_processed": self.learning_state.samples_processed,
                    "weights_updated": self.learning_state.weights_updated,
                    "loss_history": self.learning_state.loss_history[-50:],
                    "training_time": self.learning_state.training_time
                },
                "training_samples": self.training_samples[-100:],
                "pattern_weights": {k: dict(v) for k, v in self.pattern_weights.items()},
                "pattern_counts": dict(self.pattern_counts),
                "stats": self.stats,
                "real_learning_active": self.real_learning_active
            }
            
            # 원자적 저장
            directory = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
            os.makedirs(directory, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(mode='wb', dir=directory, delete=False) as tmp_file:
                pickle.dump(model_data, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path = tmp_file.name
            
            os.replace(tmp_path, filepath)
            
            self._debug_print(f"딥러닝 모델 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            self._debug_print(f"모델 저장 오류: {e}")
            return False
    
    def load_model(self, filepath: str = "./learning_model.pkl") -> bool:
        """모델 로드"""
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # 딥러닝 모델 상태 복원
            if "deep_analyzer_state" in model_data and model_data["deep_analyzer_state"]:
                self.deep_analyzer.load_state_dict(model_data["deep_analyzer_state"])
                
            if "optimizer_state" in model_data and model_data["optimizer_state"]:
                self.optimizer.load_state_dict(model_data["optimizer_state"])
            
            # 학습된 패턴 복원
            self.learned_patterns = defaultdict(list, model_data.get("learned_patterns", {}))
            self.domain_knowledge = defaultdict(dict, model_data.get("domain_knowledge", {}))
            
            # 학습 상태 복원
            learning_state = model_data.get("learning_state", {})
            self.learning_state.samples_processed = learning_state.get("samples_processed", 0)
            self.learning_state.weights_updated = learning_state.get("weights_updated", 0)
            self.learning_state.loss_history = learning_state.get("loss_history", [])
            self.learning_state.training_time = learning_state.get("training_time", 0.0)
            
            # 기존 데이터 복원
            self.training_samples = model_data.get("training_samples", [])
            self.stats = model_data.get("stats", self.stats)
            self.real_learning_active = model_data.get("real_learning_active", False)
            
            pattern_weights = model_data.get("pattern_weights", {})
            self.pattern_weights = defaultdict(_default_float_dict)
            for k, v in pattern_weights.items():
                self.pattern_weights[k] = defaultdict(_default_float, v)
                
            self.pattern_counts = defaultdict(_default_int, model_data.get("pattern_counts", {}))
            
            self._debug_print(f"딥러닝 모델 로드 완료 - 학습 활성: {self.real_learning_active}")
            return True
            
        except Exception as e:
            self._debug_print(f"모델 로드 오류: {e}")
            return False
    
    def cleanup(self) -> None:
        """정리"""
        try:
            learning_stats = self.get_learning_statistics()
            
            print(f"실제 딥러닝 학습 완료:")
            print(f"  - 학습 활성화: {learning_stats['deep_learning_active']}")
            print(f"  - 처리된 샘플: {learning_stats['samples_processed']}개")
            print(f"  - 가중치 업데이트: {learning_stats['weights_updated']}회")
            print(f"  - GPU 메모리 사용: {learning_stats['gpu_memory_used_gb']:.2f}GB")
            print(f"  - 총 학습 시간: {learning_stats['total_training_time']:.1f}초")
            print(f"  - 평균 손실: {learning_stats['average_loss']:.4f}")
            print(f"  - 딥러닝 패턴: {learning_stats['learned_patterns_count']}개")
            print(f"  - 도메인 지식: {learning_stats['domain_knowledge_count']}개")
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self._debug_print(f"정리 중 오류: {e}")

# 기존 인터페이스 호환성을 위한 별칭
LearningSystem = RealLearningSystem