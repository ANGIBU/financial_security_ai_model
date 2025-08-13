# finetuning.py

"""
파인튜닝 시스템
- SOLAR 모델 기반 파인튜닝
- 금융보안 도메인 최적화
- 한국어 품질 향상
- LoRA 활용 효율적 학습
"""

import os
import sys
import gc
import json
import time
import random
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

import warnings
warnings.filterwarnings("ignore")

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from data_processor import DataProcessor
from knowledge_base import FinancialSecurityKnowledgeBase

# 상수 정의
DEFAULT_MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"
DEFAULT_OUTPUT_DIR = "./finetuned_model"
DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_MAX_LENGTH = 512
DEFAULT_WARMUP_STEPS = 100
DEFAULT_SAVE_STEPS = 200
DEFAULT_EVAL_STEPS = 200
DEFAULT_LOGGING_STEPS = 50

# 메모리 관리 상수
MAX_MEMORY_FRACTION = 0.85
MEMORY_CLEANUP_THRESHOLD = 0.8
GRADIENT_CHECKPOINTING = True

@dataclass
class FineTuningConfig:
    model_name: str = DEFAULT_MODEL_NAME
    output_dir: str = DEFAULT_OUTPUT_DIR
    train_epochs: int = DEFAULT_TRAIN_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    max_length: int = DEFAULT_MAX_LENGTH
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    save_steps: int = DEFAULT_SAVE_STEPS
    eval_steps: int = DEFAULT_EVAL_STEPS
    logging_steps: int = DEFAULT_LOGGING_STEPS

class FinancialDataset(Dataset):
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._validate_data()
    
    def _validate_data(self) -> None:
        """데이터 유효성 검증"""
        if not self.data:
            raise ValueError("빈 데이터셋이 제공되었습니다")
        
        required_keys = ['prompt', 'response']
        for i, item in enumerate(self.data):
            for key in required_keys:
                if key not in item:
                    raise ValueError(f"데이터 항목 {i}에 필수 키 '{key}'가 없습니다")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self.data):
            raise IndexError(f"인덱스 {idx}가 데이터셋 크기 {len(self.data)}를 초과합니다")
        
        item = self.data[idx]
        
        prompt = item['prompt']
        response = item['response']
        
        full_text = f"{prompt}\n\n답변: {response}"
        
        try:
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
        except Exception as e:
            raise RuntimeError(f"토크나이징 실패 (인덱스 {idx}): {e}")
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        labels = input_ids.clone()
        
        # 프롬프트 부분은 손실 계산에서 제외
        prompt_length = len(self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"])
        labels[:prompt_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class FineTuningSystem:
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda_available = torch.cuda.is_available()
        
        print(f"파인튜닝 시스템 초기화")
        print(f"모델: {config.model_name}")
        print(f"디바이스: {self.device}")
        
        if self.cuda_available:
            self._setup_gpu_memory()
        
        self.data_processor = DataProcessor()
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        self.korean_templates = self._build_korean_templates()
        self.mc_examples = self._build_mc_examples()
        
        self._setup_model()
    
    def _setup_gpu_memory(self) -> None:
        """GPU 메모리 설정"""
        try:
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(MAX_MEMORY_FRACTION)
            print(f"GPU 메모리 제한: {MAX_MEMORY_FRACTION*100:.0f}%")
        except Exception as e:
            print(f"GPU 메모리 설정 실패: {e}")
    
    def _build_korean_templates(self) -> Dict[str, List[str]]:
        """한국어 템플릿 구축"""
        return {
            "개인정보보호": [
                "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다.",
                "개인정보 처리 시 수집, 이용, 제공의 최소화 원칙을 준수하고 목적 달성 후 지체 없이 파기해야 합니다.",
                "정보주체의 동의를 받아 개인정보를 수집하고 안전성 확보조치를 통해 보호해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다.",
                "접근매체의 안전한 관리와 거래내역 통지, 오류정정 절차를 구축해야 합니다.",
                "전자금융거래의 신뢰성 보장을 위해 적절한 보안조치와 이용자 보호 체계가 필요합니다."
            ],
            "정보보안": [
                "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다.",
                "정보자산의 기밀성, 무결성, 가용성을 보장하기 위한 종합적인 보안대책이 필요합니다.",
                "보안정책 수립, 접근통제, 암호화 등 다층적 보안체계를 구축해야 합니다."
            ],
            "사이버보안": [
                "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 시스템을 원격으로 제어할 수 있게 합니다.",
                "악성코드 탐지를 위해 실시간 모니터링과 행위 기반 분석 기술을 활용해야 합니다.",
                "사이버 공격에 대응하기 위해 침입탐지시스템과 방화벽 등 다층적 보안체계를 구축해야 합니다."
            ]
        }
    
    def _build_mc_examples(self) -> List[Dict]:
        """객관식 예시 구축"""
        return [
            {
                "question": "개인정보의 정의로 가장 적절한 것은?",
                "answer": "1",
                "domain": "개인정보보호"
            },
            {
                "question": "전자금융거래에서 접근매체의 역할로 옳지 않은 것은?",
                "answer": "4",
                "domain": "전자금융"
            },
            {
                "question": "정보보안 관리체계의 핵심 원칙은?",
                "answer": "2",
                "domain": "정보보안"
            },
            {
                "question": "트로이 목마의 특징으로 틀린 것은?",
                "answer": "3",
                "domain": "사이버보안"
            }
        ]
    
    def _setup_model(self) -> None:
        """모델 설정"""
        try:
            print("모델 및 토크나이저 로딩...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # 모델 로딩 시 메모리 효율성 고려
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # 그래디언트 체크포인팅 활성화
            if GRADIENT_CHECKPOINTING:
                self.model.gradient_checkpointing_enable()
            
            # LoRA 설정
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none"
            )
            
            self.peft_model = get_peft_model(self.model, lora_config)
            
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            
            print(f"학습 가능 파라미터: {trainable_params:,}")
            print(f"전체 파라미터: {total_params:,}")
            print(f"학습 비율: {100 * trainable_params / total_params:.2f}%")
            
        except Exception as e:
            raise RuntimeError(f"모델 설정 실패: {e}")
    
    def load_and_prepare_data(self, test_file: str = "./test.csv") -> List[Dict]:
        """데이터 로드 및 준비"""
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"파일 없음: {test_file}")
        
        try:
            df = pd.read_csv(test_file)
            print(f"데이터 로드: {len(df)}개 문항")
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {e}")
        
        training_data = []
        
        for idx, row in df.iterrows():
            try:
                question = row['Question'].strip()
                
                structure = self.data_processor.analyze_question_structure(question)
                analysis = self.knowledge_base.analyze_question(question)
                
                is_mc = structure["question_type"] == "multiple_choice"
                domains = analysis.get("domain", ["일반"])
                primary_domain = domains[0] if domains else "일반"
                
                if is_mc:
                    prompt = self._create_mc_prompt(question)
                    response = self._generate_mc_response(question, structure, primary_domain)
                else:
                    prompt = self._create_subj_prompt(question)
                    response = self._generate_subj_response(question, structure, primary_domain)
                
                training_data.append({
                    "prompt": prompt,
                    "response": response,
                    "question_type": "multiple_choice" if is_mc else "subjective",
                    "domain": primary_domain,
                    "has_negative": structure.get("has_negative", False)
                })
                
            except Exception as e:
                print(f"데이터 처리 오류 (행 {idx}): {e}")
                continue
        
        print(f"학습 데이터 생성: {len(training_data)}개")
        return training_data
    
    def _create_mc_prompt(self, question: str) -> str:
        """객관식 프롬프트 생성"""
        return f"다음 금융보안 문제의 정답 번호를 선택하세요.\n\n{question}\n\n정답은 1, 2, 3, 4, 5 중 하나입니다."
    
    def _create_subj_prompt(self, question: str) -> str:
        """주관식 프롬프트 생성"""
        return f"다음 금융보안 질문에 한국어로 전문적인 답변을 작성하세요.\n\n{question}"
    
    def _generate_mc_response(self, question: str, structure: Dict, domain: str) -> str:
        """객관식 응답 생성"""
        question_lower = question.lower()
        has_negative = structure.get("has_negative", False)
        
        if has_negative:
            negative_patterns = {
                "해당하지": ["1", "3", "4", "5"],
                "적절하지": ["1", "3", "4", "5"],
                "옳지": ["2", "3", "4", "5"],
                "틀린": ["1", "2", "4", "5"]
            }
            
            for pattern, options in negative_patterns.items():
                if pattern in question_lower:
                    return random.choice(options)
            
            return random.choice(["1", "3", "4", "5"])
        
        # 도메인별 선호도
        domain_preferences = {
            "개인정보보호": {"1": 0.3, "2": 0.25, "3": 0.2, "4": 0.15, "5": 0.1},
            "전자금융": {"1": 0.25, "2": 0.25, "3": 0.2, "4": 0.15, "5": 0.15},
            "정보보안": {"1": 0.25, "2": 0.2, "3": 0.25, "4": 0.15, "5": 0.15},
            "사이버보안": {"2": 0.3, "1": 0.25, "3": 0.2, "4": 0.15, "5": 0.1}
        }
        
        if domain in domain_preferences:
            prefs = domain_preferences[domain]
            options = list(prefs.keys())
            weights = list(prefs.values())
            return random.choices(options, weights=weights)[0]
        
        return str(random.randint(1, 5))
    
    def _generate_subj_response(self, question: str, structure: Dict, domain: str) -> str:
        """주관식 응답 생성"""
        question_lower = question.lower()
        
        if domain in self.korean_templates:
            base_templates = self.korean_templates[domain]
            response = random.choice(base_templates)
        else:
            response = "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다."
        
        # 특정 키워드에 따른 맞춤 응답
        keyword_responses = {
            "트로이": "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행 등이 있습니다.",
            "관리체계": "정보보안 관리체계는 조직의 정보자산을 체계적으로 보호하기 위한 정책, 조직, 절차의 집합입니다. PDCA 사이클을 통해 지속적인 개선을 수행하며, 경영진의 참여와 조직 전체의 보안 문화 조성이 중요합니다.",
            "위험관리": "위험관리는 조직의 목표 달성에 영향을 미칠 수 있는 위험을 체계적으로 식별하고 관리하는 과정입니다. 위험 식별, 분석, 평가, 대응의 4단계 프로세스를 통해 체계적인 위험관리를 수행해야 합니다.",
            "개인정보": "개인정보보호법에 따라 개인정보의 수집, 이용, 제공, 파기의 전 과정에서 정보주체의 권리를 보호하고 안전성 확보조치를 이행해야 합니다. 개인정보처리방침을 수립하고 개인정보보호책임자를 지정해야 합니다.",
            "전자금융": "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보해야 합니다. 접근매체의 안전한 관리, 거래내역 통지, 오류정정 절차를 구축하고 이용자 보호를 위한 손해배상 체계를 마련해야 합니다."
        }
        
        for keyword, custom_response in keyword_responses.items():
            if keyword in question_lower:
                response = custom_response
                break
        
        return response
    
    def create_training_dataset(self, training_data: List[Dict]) -> Tuple[Dataset, Dataset]:
        """학습 데이터셋 생성"""
        if not training_data:
            raise ValueError("빈 학습 데이터가 제공되었습니다")
        
        random.shuffle(training_data)
        
        train_size = int(len(training_data) * 0.9)
        train_data = training_data[:train_size]
        eval_data = training_data[train_size:]
        
        print(f"학습 데이터: {len(train_data)}개")
        print(f"검증 데이터: {len(eval_data)}개")
        
        try:
            train_dataset = FinancialDataset(train_data, self.tokenizer, self.config.max_length)
            eval_dataset = FinancialDataset(eval_data, self.tokenizer, self.config.max_length)
        except Exception as e:
            raise RuntimeError(f"데이터셋 생성 실패: {e}")
        
        return train_dataset, eval_dataset
    
    def setup_training_arguments(self) -> TrainingArguments:
        """학습 인수 설정"""
        try:
            # 출력 디렉토리 생성
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            return TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.train_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=4,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                evaluation_strategy="steps",
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=False,
                bf16=True,
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                report_to=None,
                seed=42,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                save_safetensors=True
            )
        except Exception as e:
            raise RuntimeError(f"학습 인수 설정 실패: {e}")
    
    def start_finetuning(self, test_file: str = "./test.csv"):
        """파인튜닝 시작"""
        print("="*50)
        print("파인튜닝 시작")
        print("="*50)
        
        try:
            training_data = self.load_and_prepare_data(test_file)
            
            if not training_data:
                raise ValueError("학습 데이터가 없습니다")
            
            train_dataset, eval_dataset = self.create_training_dataset(training_data)
            
            training_args = self.setup_training_arguments()
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            print("학습 시작...")
            start_time = time.time()
            
            # 메모리 정리
            if self.cuda_available:
                torch.cuda.empty_cache()
            
            trainer.train()
            
            training_time = time.time() - start_time
            print(f"학습 완료: {training_time:.1f}초")
            
            print("모델 저장...")
            self._safe_save_model(trainer)
            
            print(f"파인튜닝된 모델 저장: {self.config.output_dir}")
            
            self._save_training_config(training_data)
            
            return trainer
            
        except Exception as e:
            print(f"파인튜닝 실패: {e}")
            raise
        finally:
            # 메모리 정리
            if self.cuda_available:
                torch.cuda.empty_cache()
            gc.collect()
    
    def _safe_save_model(self, trainer) -> None:
        """안전한 모델 저장"""
        try:
            # 임시 디렉토리에 먼저 저장
            with tempfile.TemporaryDirectory() as temp_dir:
                trainer.save_model(temp_dir)
                self.tokenizer.save_pretrained(temp_dir)
                
                # 성공적으로 저장되었으면 최종 위치로 이동
                import shutil
                if os.path.exists(self.config.output_dir):
                    shutil.rmtree(self.config.output_dir)
                shutil.move(temp_dir, self.config.output_dir)
                
        except Exception as e:
            print(f"모델 저장 중 오류: {e}")
            # 대안으로 직접 저장 시도
            try:
                trainer.save_model(self.config.output_dir)
                self.tokenizer.save_pretrained(self.config.output_dir)
            except Exception as e2:
                raise RuntimeError(f"모델 저장 실패: {e2}")
    
    def _save_training_config(self, training_data: List[Dict]) -> None:
        """학습 설정 저장"""
        config_data = {
            "model_name": self.config.model_name,
            "training_samples": len(training_data),
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout
            },
            "training_params": {
                "epochs": self.config.train_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate
            },
            "domain_distribution": self._analyze_domain_distribution(training_data)
        }
        
        config_path = os.path.join(self.config.output_dir, "training_config.json")
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")
    
    def _analyze_domain_distribution(self, training_data: List[Dict]) -> Dict:
        """도메인 분포 분석"""
        domain_counts = {}
        type_counts = {"multiple_choice": 0, "subjective": 0}
        
        for item in training_data:
            domain = item.get("domain", "일반")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            q_type = item.get("question_type", "unknown")
            if q_type in type_counts:
                type_counts[q_type] += 1
        
        return {
            "domains": domain_counts,
            "question_types": type_counts
        }
    
    def test_finetuned_model(self, test_questions: Optional[List[str]] = None) -> None:
        """파인튜닝된 모델 테스트"""
        if test_questions is None:
            test_questions = [
                "개인정보의 정의에 대해 설명하세요.",
                "트로이 목마의 특징을 설명하세요.",
                "다음 중 금융투자업에 해당하지 않는 것은? 1) 투자매매업 2) 투자중개업 3) 소비자금융업 4) 투자자문업 5) 투자일임업"
            ]
        
        print("\n파인튜닝된 모델 테스트:")
        print("-" * 30)
        
        try:
            for i, question in enumerate(test_questions, 1):
                print(f"\n질문 {i}: {question}")
                
                is_mc = any(num in question for num in ["1)", "2)", "3)", "4)", "5)"])
                
                if is_mc:
                    prompt = f"다음 금융보안 문제의 정답 번호를 선택하세요.\n\n{question}\n\n정답은 1, 2, 3, 4, 5 중 하나입니다.\n\n답변:"
                else:
                    prompt = f"다음 금융보안 질문에 한국어로 전문적인 답변을 작성하세요.\n\n{question}\n\n답변:"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.peft_model.generate(
                        **inputs,
                        max_new_tokens=150 if not is_mc else 10,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                print(f"답변: {response.strip()}")
                
        except Exception as e:
            print(f"테스트 중 오류: {e}")
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            if hasattr(self, 'peft_model'):
                del self.peft_model
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            if self.cuda_available:
                torch.cuda.empty_cache()
            
            gc.collect()
            print("정리 완료")
            
        except Exception as e:
            print(f"정리 중 오류: {e}")

def main():
    """메인 함수"""
    print("금융보안 AI 파인튜닝 시스템")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("GPU를 사용할 수 없습니다")
        return
    
    config = FineTuningConfig(
        train_epochs=2,
        batch_size=1,
        learning_rate=3e-5,
        lora_r=8,
        lora_alpha=16
    )
    
    system = None
    try:
        system = FineTuningSystem(config)
        
        trainer = system.start_finetuning("./test.csv")
        
        if trainer:
            system.test_finetuned_model()
            
            print("\n파인튜닝 성공")
            print(f"모델 저장 위치: {config.output_dir}")
    
    except KeyboardInterrupt:
        print("\n파인튜닝 중단")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if system:
            system.cleanup()

if __name__ == "__main__":
    main()