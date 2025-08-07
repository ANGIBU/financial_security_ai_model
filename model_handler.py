# model_handler.py
"""
모델 핸들러
"""

import torch
import re
import time
import gc
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
import warnings
warnings.filterwarnings("ignore")

@dataclass
class InferenceResult:
    """추론 결과"""
    response: str
    confidence: float
    reasoning_quality: float
    analysis_depth: int
    inference_time: float = 0.0

class ModelHandler:
    """모델 핸들러"""
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 load_in_4bit: bool = False, max_memory_gb: int = 22):
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb
        
        print(f"모델 로딩: {model_name}")
        
        # 토크나이저 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 모델 설정
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        
        # 4bit 양자화 설정
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        
        # Flash Attention 지원 확인
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Flash Attention 2 활성화")
        except ImportError:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                model_kwargs["attn_implementation"] = "sdpa"
                print("SDPA Attention 활성화")
            else:
                print("Standard Attention 사용")
        
        # 모델 로딩
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # 평가 모드
        self.model.eval()
        
        # 생성 설정
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_new_tokens=512,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # 캐시
        self.response_cache = {}
        self.cache_hits = 0
        
        print("모델 로딩 완료")
    
    def _filter_korean_only(self, text: str) -> str:
        """한국어만 유지하고 다른 언어 문자 제거"""
        
        # 한자 및 기타 언어 문자를 한국어로 교체하거나 제거
        text = re.sub(r'[軟软][件体]', '소프트웨어', text)
        text = re.sub(r'[危険]害', '위험', text)
        text = re.sub(r'可能性', '가능성', text)
        text = re.sub(r'[存在]', '존재', text)
        text = re.sub(r'程[式序]', '프로그램', text)
        text = re.sub(r'金融', '금융', text)
        text = re.sub(r'交易', '거래', text)
        text = re.sub(r'安全', '안전', text)
        text = re.sub(r'保險', '보험', text)
        text = re.sub(r'方案', '방안', text)
        
        # 괄호 밖의 단독 영어 단어 제거 (객관식 제외)
        if not re.match(r'^\s*[1-5]\s*$', text.strip()):
            text = re.sub(r'\b[A-Za-z]+\b(?!\))', '', text)
        
        # 남은 한자 문자 제거
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        
        # 중복 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _validate_korean_response(self, response: str, question_type: str) -> bool:
        """한국어 응답 검증 (객관식은 관대하게)"""
        
        # 객관식은 숫자만 있으면 통과
        if question_type == "multiple_choice":
            if re.match(r'^\s*[1-5]\s*$', response.strip()):
                return True
            # 숫자가 포함되어 있으면 관대하게 허용
            if re.search(r'[1-5]', response):
                return True
        
        # 한자 문자 확인
        if re.search(r'[\u4e00-\u9fff]', response):
            return False
        
        # 주관식만 엄격한 한국어 비율 확인
        if question_type != "multiple_choice":
            korean_chars = len(re.findall(r'[가-힣]', response))
            total_chars = len(re.sub(r'[^\w]', '', response))
            
            if total_chars > 0:
                korean_ratio = korean_chars / total_chars
                if korean_ratio < 0.5:  # 50%로 완화
                    return False
        
        return True
    
    def generate_response(self, prompt: str, question_type: str,
                         max_attempts: int = 2) -> InferenceResult:
        """응답 생성"""
        
        start_time = time.time()
        
        # 캐시 확인
        cache_key = hash(prompt[:200])
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached = self.response_cache[cache_key]
            cached.inference_time = 0.01
            return cached
        
        best_result = None
        best_score = 0
        
        for attempt in range(max_attempts):
            try:
                # 객관식: 더 보수적 설정
                if question_type == "multiple_choice":
                    gen_config = GenerationConfig(
                        do_sample=False if attempt == 0 else True,
                        temperature=0.1 if attempt == 0 else 0.3,
                        top_p=0.7,
                        top_k=20,
                        max_new_tokens=64,
                        repetition_penalty=1.02,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # 주관식: 품질 우선
                    gen_config = GenerationConfig(
                        do_sample=True,
                        temperature=0.4,
                        top_p=0.85,
                        top_k=40,
                        max_new_tokens=400,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # 토크나이징
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
                
                # 생성
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=gen_config,
                            use_cache=True
                        )
                
                # 디코딩
                raw_response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # 한국어 필터링 적용 (주관식만)
                if question_type != "multiple_choice":
                    filtered_response = self._filter_korean_only(raw_response)
                else:
                    filtered_response = raw_response
                
                # 한국어 검증 (관대하게)
                if not self._validate_korean_response(filtered_response, question_type):
                    continue
                
                # 응답 평가
                result = self._evaluate_response(filtered_response, question_type)
                result.inference_time = time.time() - start_time
                
                # 최고 품질 선택
                score = result.confidence * result.reasoning_quality
                if score > best_score:
                    best_score = score
                    best_result = result
                
                # 충분히 좋으면 조기 종료
                if score > 0.6:
                    break
                    
            except Exception as e:
                continue
        
        # 실패 시 폴백
        if best_result is None:
            if question_type == "multiple_choice":
                best_result = InferenceResult(
                    response="3",
                    confidence=0.3,
                    reasoning_quality=0.3,
                    analysis_depth=1,
                    inference_time=time.time() - start_time
                )
            else:
                best_result = InferenceResult(
                    response="관련 규정에 따른 적절한 조치가 필요합니다.",
                    confidence=0.3,
                    reasoning_quality=0.3,
                    analysis_depth=1,
                    inference_time=time.time() - start_time
                )
        
        # 캐시 저장
        if best_score > 0.4:
            self.response_cache[cache_key] = best_result
        
        return best_result
    
    def _evaluate_response(self, response: str, question_type: str) -> InferenceResult:
        """응답 평가"""
        
        if question_type == "multiple_choice":
            confidence = 0.5
            reasoning = 0.5
            
            # 답변 패턴 확인
            if re.search(r'^[1-5]$', response.strip()):
                confidence += 0.4
            elif re.search(r'[1-5]', response):
                confidence += 0.2
            
            # 깔끔한 답변 보너스
            if len(response.strip()) <= 3:
                reasoning += 0.3
            
            return InferenceResult(
                response=response,
                confidence=min(confidence, 1.0),
                reasoning_quality=min(reasoning, 1.0),
                analysis_depth=2
            )
        
        else:
            confidence = 0.4
            reasoning = 0.4
            
            # 길이 체크
            length = len(response)
            if 100 <= length <= 800:
                confidence += 0.3
            elif 50 <= length < 100:
                confidence += 0.1
            
            # 전문 용어
            keywords = ['법', '규정', '보안', '관리', '조치', '정책', '체계']
            keyword_count = sum(1 for k in keywords if k in response)
            if keyword_count >= 3:
                confidence += 0.2
                reasoning += 0.3
            elif keyword_count >= 1:
                confidence += 0.1
                reasoning += 0.1
            
            # 구조화
            if re.search(r'첫째|둘째|1\)|2\)', response):
                reasoning += 0.2
            
            # 한국어 품질 보너스
            korean_ratio = len(re.findall(r'[가-힣]', response)) / max(len(response), 1)
            if korean_ratio > 0.7:
                confidence += 0.1
            
            return InferenceResult(
                response=response,
                confidence=min(confidence, 1.0),
                reasoning_quality=min(reasoning, 1.0),
                analysis_depth=3
            )
    
    def generate_batch(self, prompts: List[str], question_types: List[str],
                      batch_size: int = 8) -> List[InferenceResult]:
        """배치 처리"""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_types = question_types[i:i+batch_size]
            
            for prompt, q_type in zip(batch_prompts, batch_types):
                result = self.generate_response(prompt, q_type, max_attempts=1)
                results.append(result)
            
            # 메모리 정리
            if (i + batch_size) % 32 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def cleanup(self):
        """메모리 정리"""
        print(f"캐시 히트: {self.cache_hits}회")
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        self.response_cache.clear()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict:
        """모델 정보"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache)
        }