# model_handler.py
"""
모델 핸들러
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
import gc
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import threading
from queue import Queue

@dataclass
class InferenceResult:
    """추론 결과 데이터 클래스"""
    response: str
    confidence: float
    reasoning_quality: float
    analysis_depth: int
    inference_time: float = 0.0

class OptimizedModelHandler:
    """모델 핸들러"""
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 load_in_4bit: bool = False, max_memory_gb: int = 22):
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb
        
        print(f"모델 로딩: {model_name}")
        
        # RTX 4090 24GB - 16bit 정밀도 사용
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quantization_config = None
        
        # 토크나이저 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로딩
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_memory={0: f"{max_memory_gb}GB"},
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        
        # 추론 설정
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        
        # torch.compile 사용
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("모델 컴파일 완료")
            except:
                print("모델 컴파일 스킵")
        
        # 파이프라인 설정
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            return_full_text=False,
            batch_size=1,
            torch_dtype=torch.float16
        )
        
        # 캐시 설정
        self.response_cache = {}
        self.cache_hits = 0
        
        print("모델 로딩 완료")
    
    def generate_expert_response(self, prompt: str, question_type: str,
                                max_attempts: int = 2) -> InferenceResult:
        """응답 생성"""
        
        start_time = time.time()
        
        # 캐시 확인
        cache_key = hash(prompt[:100])
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached_result = self.response_cache[cache_key]
            cached_result.inference_time = 0.01
            return cached_result
        
        best_result = None
        best_score = 0
        
        for attempt in range(max_attempts):
            try:
                # 시도별 파라미터 조정
                generation_params = self._get_optimized_params(question_type, attempt)
                
                # 추론 실행
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        outputs = self.pipe(prompt, **generation_params)
                        response = outputs[0]["generated_text"].strip()
                
                # 응답 품질 평가
                result = self._evaluate_response_quality(response, question_type)
                result.inference_time = time.time() - start_time
                
                # 최고 품질 응답 선택
                current_score = result.confidence * result.reasoning_quality
                if current_score > best_score:
                    best_score = current_score
                    best_result = result
                
                # 충분히 좋은 응답이면 조기 종료
                if current_score > 0.7 or (attempt == 0 and current_score > 0.5):
                    break
                    
            except Exception as e:
                continue
        
        # 실패 시 기본 응답
        if best_result is None:
            if question_type == "multiple_choice":
                best_result = InferenceResult(
                    response="분석 결과 2번이 가장 적절합니다.",
                    confidence=0.3,
                    reasoning_quality=0.3,
                    analysis_depth=1,
                    inference_time=time.time() - start_time
                )
            else:
                best_result = InferenceResult(
                    response="해당 사항은 금융보안 규정에 따라 적절한 조치가 필요합니다.",
                    confidence=0.3,
                    reasoning_quality=0.3,
                    analysis_depth=1,
                    inference_time=time.time() - start_time
                )
        
        # 캐시 저장
        if best_score > 0.6:
            self.response_cache[cache_key] = best_result
        
        return best_result
    
    def _get_optimized_params(self, question_type: str, attempt: int) -> Dict:
        """생성 파라미터"""
        
        if question_type == "multiple_choice":
            # 객관식: 빠르고 정확한 생성
            if attempt == 0:
                base_params = {
                    "max_new_tokens": 256,
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 30,
                    "do_sample": True,
                    "repetition_penalty": 1.05,
                    "no_repeat_ngram_size": 2,
                }
            else:
                base_params = {
                    "max_new_tokens": 512,
                    "temperature": 0.2,
                    "top_p": 0.85,
                    "top_k": 40,
                    "do_sample": True,
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 2,
                }
        else:
            # 주관식: 완성도 우선
            base_params = {
                "max_new_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 3,
            }
        
        # 공통 파라미터
        base_params.update({
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "early_stopping": True,
        })
        
        return base_params
    
    def generate_batch_responses(self, prompts: List[str], question_types: List[str], 
                               batch_size: int = 10) -> List[InferenceResult]:
        """배치 추론"""
        results = []
        
        # 동일 타입끼리 그룹화
        type_groups = {}
        for i, (prompt, q_type) in enumerate(zip(prompts, question_types)):
            if q_type not in type_groups:
                type_groups[q_type] = []
            type_groups[q_type].append((i, prompt))
        
        # 타입별 배치 처리
        for q_type, group in type_groups.items():
            indices = [g[0] for g in group]
            group_prompts = [g[1] for g in group]
            
            # 배치로 나누어 처리
            for i in range(0, len(group_prompts), batch_size):
                batch_prompts = group_prompts[i:i+batch_size]
                
                try:
                    # 배치 추론
                    batch_results = self._process_batch_fast(batch_prompts, q_type)
                    
                    # 결과를 원래 순서대로 저장
                    for j, result in enumerate(batch_results):
                        original_idx = indices[i+j]
                        results.append((original_idx, result))
                        
                except Exception as e:
                    print(f"배치 처리 실패, 개별 처리로 전환: {e}")
                    # 실패한 배치는 개별 처리
                    for j, prompt in enumerate(batch_prompts):
                        original_idx = indices[i+j]
                        result = self.generate_expert_response(prompt, q_type, max_attempts=1)
                        results.append((original_idx, result))
                
                # 메모리 정리
                if len(results) % 50 == 0:
                    torch.cuda.empty_cache()
        
        # 원래 순서로 정렬
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def _process_batch_fast(self, prompts: List[str], question_type: str) -> List[InferenceResult]:
        """빠른 배치 처리"""
        generation_params = self._get_optimized_params(question_type, 0)
        
        # 토큰화
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # 배치 생성
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    **inputs,
                    **generation_params,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        # 디코딩
        responses = self.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 결과 평가
        results = []
        for response in responses:
            result = self._evaluate_response_quality(response.strip(), question_type)
            results.append(result)
        
        return results
    
    def _evaluate_response_quality(self, response: str, question_type: str) -> InferenceResult:
        """응답 품질 평가"""
        
        if question_type == "multiple_choice":
            return self._evaluate_mc_response_fast(response)
        else:
            return self._evaluate_subjective_response_fast(response)
    
    def _evaluate_mc_response_fast(self, response: str) -> InferenceResult:
        """객관식 응답 평가"""
        confidence = 0.5
        reasoning_quality = 0.5
        
        # 답변 번호 추출 가능성
        answer_patterns = [
            r'정답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'([1-5])번',
            r'선택지\s*([1-5])',
        ]
        
        for pattern in answer_patterns:
            if re.search(pattern, response):
                confidence += 0.3
                break
        
        # 분석 키워드 존재
        if any(keyword in response for keyword in ['분석', '근거', '따라서', '법']):
            reasoning_quality += 0.2
        
        # 길이 체크
        if 50 <= len(response) <= 800:
            confidence += 0.1
        
        return InferenceResult(
            response=response,
            confidence=min(confidence, 1.0),
            reasoning_quality=min(reasoning_quality, 1.0),
            analysis_depth=2
        )
    
    def _evaluate_subjective_response_fast(self, response: str) -> InferenceResult:
        """주관식 응답 평가"""
        confidence = 0.5
        reasoning_quality = 0.5
        
        # 길이 체크
        response_length = len(response)
        if 100 <= response_length <= 1000:
            confidence += 0.2
        
        # 전문 용어 체크
        if any(term in response for term in ['보안', '개인정보', '금융', '시스템']):
            confidence += 0.2
            reasoning_quality += 0.2
        
        return InferenceResult(
            response=response,
            confidence=min(confidence, 1.0),
            reasoning_quality=min(reasoning_quality, 1.0),
            analysis_depth=3
        )
    
    def cleanup(self):
        """메모리 정리"""
        # 캐시 통계 출력
        if self.response_cache:
            print(f"캐시 히트: {self.cache_hits}회")
        
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "memory_usage": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "max_memory": self.max_memory_gb,
            "cache_hits": self.cache_hits
        }