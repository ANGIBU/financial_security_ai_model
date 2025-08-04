# model_handler.py
"""
최적화된 모델 핸들러
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

@dataclass
class InferenceResult:
    """추론 결과 데이터 클래스"""
    response: str
    confidence: float
    reasoning_quality: float
    analysis_depth: int

class OptimizedModelHandler:
    """최적화된 모델 핸들러"""
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 load_in_4bit: bool = False, max_memory_gb: int = 22):  # RTX 4090 기본값
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb
        
        print(f"모델 로딩: {model_name}")
        
        # RTX 4090 24GB - 16bit 정밀도 사용 가능
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
        
        # 모델 로딩 (RTX 4090 24GB 최적화)
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
        
        # 추론 최적화 설정
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        
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
        
        print("모델 로딩 완료")
    
    def generate_expert_response(self, prompt: str, question_type: str,
                                max_attempts: int = 3) -> InferenceResult:
        """전문가급 응답 생성 (다단계 검증)"""
        
        best_result = None
        best_score = 0
        
        for attempt in range(max_attempts):
            try:
                # 시도별 파라미터 조정
                generation_params = self._get_generation_params(question_type, attempt)
                
                # 추론 실행
                with torch.no_grad():
                    outputs = self.pipe(prompt, **generation_params)
                    response = outputs[0]["generated_text"].strip()
                
                # 응답 품질 평가
                result = self._evaluate_response_quality(response, question_type)
                
                # 최고 품질 응답 선택
                current_score = result.confidence * result.reasoning_quality
                if current_score > best_score:
                    best_score = current_score
                    best_result = result
                
                # 충분히 좋은 응답이면 조기 종료
                if current_score > 0.8:
                    break
                    
            except Exception as e:
                continue
        
        # 모든 시도 실패 시 기본 응답
        if best_result is None:
            if question_type == "multiple_choice":
                best_result = InferenceResult(
                    response="분석이 어려워 추가 검토가 필요합니다. 1",
                    confidence=0.1,
                    reasoning_quality=0.1,
                    analysis_depth=0
                )
            else:
                best_result = InferenceResult(
                    response="해당 문제에 대한 전문적 분석이 필요한 복잡한 사안입니다.",
                    confidence=0.1,
                    reasoning_quality=0.1,
                    analysis_depth=0
                )
        
        return best_result
    
    def _get_generation_params(self, question_type: str, attempt: int) -> Dict:
        """질문 유형과 시도 횟수에 따른 최적 파라미터"""
        
        if question_type == "multiple_choice":
            # 객관식: 정확성 우선
            base_params = {
                "max_new_tokens": 512,  # 충분한 분석 공간
                "temperature": 0.1 + (attempt * 0.05),  # 시도별 다양성 증가
                "top_p": 0.85,
                "top_k": 40,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "length_penalty": 0.5,  # 적절한 길이 선호
                "no_repeat_ngram_size": 2,
            }
        else:
            # 주관식: 논리성과 완성도 우선
            base_params = {
                "max_new_tokens": 768,  # 더 긴 설명 허용
                "temperature": 0.2 + (attempt * 0.1),
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,
            }
        
        # 공통 파라미터
        base_params.update({
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "early_stopping": True,
        })
        
        return base_params
    
    def _evaluate_response_quality(self, response: str, question_type: str) -> InferenceResult:
        """응답 품질 평가"""
        
        if question_type == "multiple_choice":
            return self._evaluate_mc_response(response)
        else:
            return self._evaluate_subjective_response(response)
    
    def _evaluate_mc_response(self, response: str) -> InferenceResult:
        """객관식 응답 평가"""
        confidence = 0.0
        reasoning_quality = 0.0
        analysis_depth = 0
        
        # 답변 번호 추출 가능성
        answer_patterns = [
            r'정답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'([1-5])번',
            r'^([1-5])$',
            r'선택지\s*([1-5])',
            r'결론.*?([1-5])'
        ]
        
        has_clear_answer = False
        for pattern in answer_patterns:
            if re.search(pattern, response):
                has_clear_answer = True
                confidence += 0.3
                break
        
        # 분석 과정 존재 여부
        analysis_indicators = [
            '분석', '검토', '근거', '이유', '판단', '따라서',
            '법령', '규정', '원칙', '정의', '조항'
        ]
        
        analysis_count = sum(1 for indicator in analysis_indicators if indicator in response)
        analysis_depth = min(analysis_count, 5)
        reasoning_quality = min(analysis_count / 10, 1.0)
        
        # 전문적 용어 사용
        expert_terms = [
            '개인정보보호법', '전자금융거래법', '정보통신망법',
            '암호화', '접근제어', '보안', '인증', '권한'
        ]
        
        expert_term_count = sum(1 for term in expert_terms if term in response)
        confidence += min(expert_term_count * 0.1, 0.3)
        
        # 논리적 구조
        structure_indicators = ['단계', '첫째', '둘째', '따라서', '결론']
        structure_score = sum(1 for indicator in structure_indicators if indicator in response)
        reasoning_quality += min(structure_score * 0.1, 0.3)
        
        # 길이 적절성 (너무 짧거나 길지 않은지)
        response_length = len(response)
        if 100 <= response_length <= 1000:
            confidence += 0.2
        
        return InferenceResult(
            response=response,
            confidence=min(confidence, 1.0),
            reasoning_quality=min(reasoning_quality, 1.0),
            analysis_depth=analysis_depth
        )
    
    def _evaluate_subjective_response(self, response: str) -> InferenceResult:
        """주관식 응답 평가"""
        confidence = 0.0
        reasoning_quality = 0.0
        analysis_depth = 0
        
        # 길이 적절성
        response_length = len(response)
        if 200 <= response_length <= 1500:
            confidence += 0.3
        elif 100 <= response_length < 200:
            confidence += 0.1
        
        # 전문적 내용
        professional_terms = [
            '개인정보', '보안', '금융', '관리', '시스템', '정책',
            '절차', '조치', '방안', '체계', '원칙', '기준'
        ]
        
        term_count = sum(1 for term in professional_terms if term in response)
        confidence += min(term_count * 0.05, 0.4)
        
        # 구조적 완성도
        structure_elements = [
            '정의', '개념', '방법', '절차', '고려사항', '중요',
            '필요', '포함', '구성', '요소'
        ]
        
        structure_count = sum(1 for element in structure_elements if element in response)
        reasoning_quality = min(structure_count * 0.1, 1.0)
        analysis_depth = min(structure_count, 8)
        
        # 논리적 연결성
        connectors = ['또한', '따라서', '그러므로', '이를 통해', '결과적으로']
        connector_count = sum(1 for connector in connectors if connector in response)
        reasoning_quality += min(connector_count * 0.1, 0.2)
        
        return InferenceResult(
            response=response,
            confidence=min(confidence, 1.0),
            reasoning_quality=min(reasoning_quality, 1.0),
            analysis_depth=analysis_depth
        )
    
    def generate_batch_responses(self, prompts: List[str], question_types: List[str], 
                               batch_size: int = 8) -> List[InferenceResult]:  # RTX 4090 기본 배치 크기 증가
        """배치 추론 (메모리 효율적)"""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_types = question_types[i:i+batch_size]
            
            # 같은 타입만 배치 처리
            if len(set(batch_types)) == 1:
                try:
                    batch_results = self._process_batch(batch_prompts, batch_types[0])
                    results.extend(batch_results)
                except Exception:
                    # 배치 실패 시 개별 처리
                    for prompt, q_type in zip(batch_prompts, batch_types):
                        result = self.generate_expert_response(prompt, q_type, max_attempts=1)
                        results.append(result)
            else:
                # 개별 처리
                for prompt, q_type in zip(batch_prompts, batch_types):
                    result = self.generate_expert_response(prompt, q_type, max_attempts=1)
                    results.append(result)
            
            # 메모리 정리
            if i % 20 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def _process_batch(self, prompts: List[str], question_type: str) -> List[InferenceResult]:
        """동일 타입 배치 처리"""
        generation_params = self._get_generation_params(question_type, 0)
        generation_params["batch_size"] = len(prompts)
        
        with torch.no_grad():
            batch_outputs = self.pipe(prompts, **generation_params)
            
        results = []
        for output in batch_outputs:
            response = output[0]["generated_text"].strip()
            result = self._evaluate_response_quality(response, question_type)
            results.append(result)
        
        return results
    
    def generate_with_verification(self, prompt: str, question_type: str,
                                  target_confidence: float = 0.7) -> InferenceResult:
        """검증을 통한 고품질 응답 생성"""
        
        max_attempts = 5
        for attempt in range(max_attempts):
            result = self.generate_expert_response(prompt, question_type, max_attempts=1)
            
            # 목표 신뢰도 달성 시 반환
            if result.confidence >= target_confidence:
                return result
            
            # 마지막 시도가 아니면 프롬프트 수정하여 재시도
            if attempt < max_attempts - 1:
                prompt = self._enhance_prompt_for_retry(prompt, result, attempt)
        
        return result
    
    def _enhance_prompt_for_retry(self, original_prompt: str, 
                                 previous_result: InferenceResult, attempt: int) -> str:
        """재시도를 위한 프롬프트 강화"""
        
        enhancements = [
            "더 정확하고 세밀한 분석을 통해",
            "법적 근거를 명확히 제시하여",
            "단계별로 논리적으로 접근하여",
            "전문가적 관점에서 종합적으로"
        ]
        
        enhancement = enhancements[attempt % len(enhancements)]
        
        # 프롬프트 앞부분에 강화 지시문 추가
        if "당신은" in original_prompt:
            parts = original_prompt.split("당신은", 1)
            enhanced_prompt = f"{parts[0]}당신은 {enhancement} {parts[1]}"
        else:
            enhanced_prompt = f"{enhancement} 분석해주세요.\n\n{original_prompt}"
        
        return enhanced_prompt
    
    def cleanup(self):
        """메모리 정리"""
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
            "max_memory": self.max_memory_gb
        }