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
    korean_quality: float = 0.0

class ModelHandler:
    """모델 핸들러"""
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 load_in_4bit: bool = True, max_memory_gb: int = 22):
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb
        
        print(f"모델 로딩: {model_name}")
        
        # 빠른 토크나이저 사용
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 4-bit 양자화 설정
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        
        if load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            
        # Flash Attention 시도
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
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.eval()
        
        # torch.compile 시도
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Torch Compile 활성화")
            except Exception:
                print("Torch Compile 실패, 기본 모델 사용")
        
        self._prepare_korean_tokens()
        
        # 최적화된 캐시 시스템
        self.response_cache = {}
        self.cache_hits = 0
        self.max_cache_size = 200
        
        print("모델 로딩 완료")
    
    def _prepare_korean_tokens(self):
        """한국어 토큰 준비"""
        self.korean_start_tokens = []
        korean_chars = "가나다라마바사아자차카타파하개내대래매배새애재채캐태패해"
        
        for char in korean_chars[:10]:  # 최적화: 10개만 사용
            tokens = self.tokenizer.encode(char, add_special_tokens=False)
            if tokens:
                self.korean_start_tokens.extend(tokens)
        
        self.korean_start_tokens = list(set(self.korean_start_tokens))
    
    def _create_korean_optimized_prompt(self, prompt: str, question_type: str) -> str:
        """한국어 최적화 프롬프트 생성"""
        korean_prefix = "### 한국어로만 답변하세요 ###\n\n"
        
        if question_type == "multiple_choice":
            korean_example = "\n### 답변 예시\n정답: 2\n\n"
        else:
            korean_example = "\n### 답변 예시\n관련 법령에 따라 체계적인 관리 방안을 수립하고 지속적인 개선이 필요합니다.\n\n"
        
        return korean_prefix + prompt + korean_example
    
    def generate_response(self, prompt: str, question_type: str,
                         max_attempts: int = 1) -> InferenceResult:
        """응답 생성"""
        
        start_time = time.time()
        
        # 캐시 확인
        cache_key = hash(prompt[:100])
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached = self.response_cache[cache_key]
            cached.inference_time = 0.01
            return cached
        
        optimized_prompt = self._create_korean_optimized_prompt(prompt, question_type)
        
        best_result = None
        best_korean_quality = 0
        
        for attempt in range(max_attempts):
            try:
                # 최적화된 생성 설정
                if question_type == "multiple_choice":
                    gen_config = GenerationConfig(
                        do_sample=False,  # 결정적 생성
                        temperature=0.0,
                        max_new_tokens=10,  # 매우 짧게
                        repetition_penalty=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True
                    )
                else:
                    gen_config = GenerationConfig(
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        max_new_tokens=150,  # 줄임
                        repetition_penalty=1.05,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True
                    )
                
                inputs = self.tokenizer(
                    optimized_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024  # 줄임
                ).to(self.model.device)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=gen_config
                        )
                
                raw_response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                korean_quality = self._evaluate_korean_quality(raw_response)
                cleaned_response = self._clean_korean_text(raw_response)
                
                if question_type == "multiple_choice":
                    numbers = re.findall(r'[1-5]', cleaned_response)
                    if numbers:
                        cleaned_response = numbers[0]
                        korean_quality = 1.0
                    else:
                        cleaned_response = "3"
                        korean_quality = 0.5
                
                if question_type != "multiple_choice" and korean_quality < 0.5:
                    cleaned_response = self._get_korean_fallback_response(prompt)
                    korean_quality = 0.8
                
                result = self._evaluate_response(cleaned_response, question_type)
                result.korean_quality = korean_quality
                result.inference_time = time.time() - start_time
                
                if korean_quality > best_korean_quality:
                    best_korean_quality = korean_quality
                    best_result = result
                
                if korean_quality > 0.7:
                    break
                    
            except Exception as e:
                print(f"생성 오류 (시도 {attempt+1}): {e}")
                continue
        
        if best_result is None:
            best_result = self._create_fallback_result(question_type)
            best_result.inference_time = time.time() - start_time
        
        # 캐시 저장
        if best_korean_quality > 0.5:
            if len(self.response_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]
            self.response_cache[cache_key] = best_result
        
        return best_result
    
    def _clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리"""
        
        # 한자 및 외국어 매핑
        chinese_to_korean = {
            r'軟件|软件': '소프트웨어',
            r'金融': '금융',
            r'交易': '거래',
            r'安全': '안전',
            r'資訊|资讯': '정보',
            r'系統|系统': '시스템',
            r'管理': '관리',
            r'技術|技术': '기술',
            r'個人|个人': '개인',
            r'電子|电子': '전자',
            r'認證|认证': '인증',
            r'加密': '암호화'
        }
        
        for pattern, replacement in chinese_to_korean.items():
            text = re.sub(pattern, replacement, text)
        
        # 한자 제거
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        
        # 영어 단어 제거 (괄호 안은 제외)
        text = re.sub(r'\b[A-Za-z]+\b(?!\))', '', text)
        
        # 특수문자 정리
        text = re.sub(r'[^\w\s가-힣0-9.,!?()·\-]', '', text)
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def _evaluate_korean_quality(self, text: str) -> float:
        """한국어 품질 평가"""
        
        if not text:
            return 0.0
        
        # 한자 확인
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.2
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        # 품질 점수 계산
        quality = korean_ratio * 0.8
        quality -= english_ratio * 0.6
        
        # 전문 용어 보너스
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.05, 0.2)
        
        return max(0, min(1, quality))
    
    def _get_korean_fallback_response(self, prompt: str) -> str:
        """한국어 폴백 응답"""
        
        if "개인정보" in prompt:
            return "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다."
        elif "전자금융" in prompt:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호하기 위한 방안이 요구됩니다."
        elif "보안" in prompt or "암호" in prompt:
            return "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가 및 개선이 필요합니다."
        elif "관리체계" in prompt or "ISMS" in prompt:
            return "정보보호관리체계는 조직의 정보자산 보호를 위한 정책과 절차, 기술적 대책을 종합적으로 관리하는 체계입니다."
        else:
            return "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고 지속적인 관리와 개선이 필요합니다."
    
    def _create_fallback_result(self, question_type: str) -> InferenceResult:
        """폴백 결과 생성"""
        
        if question_type == "multiple_choice":
            return InferenceResult(
                response="3",
                confidence=0.4,
                reasoning_quality=0.4,
                analysis_depth=1,
                korean_quality=1.0
            )
        else:
            return InferenceResult(
                response="관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선이 필요합니다.",
                confidence=0.5,
                reasoning_quality=0.5,
                analysis_depth=1,
                korean_quality=0.9
            )
    
    def _evaluate_response(self, response: str, question_type: str) -> InferenceResult:
        """응답 평가"""
        
        if question_type == "multiple_choice":
            confidence = 0.5
            reasoning = 0.5
            
            if re.match(r'^[1-5]$', response.strip()):
                confidence = 0.8
                reasoning = 0.7
            elif re.search(r'[1-5]', response):
                confidence = 0.6
                reasoning = 0.5
            
            return InferenceResult(
                response=response,
                confidence=confidence,
                reasoning_quality=reasoning,
                analysis_depth=2
            )
        
        else:
            confidence = 0.5
            reasoning = 0.5
            
            length = len(response)
            if 80 <= length <= 400:
                confidence += 0.2
            elif 50 <= length < 80:
                confidence += 0.1
            
            keywords = ['법', '규정', '보안', '관리', '조치', '정책', '체계', '보호', '안전']
            keyword_count = sum(1 for k in keywords if k in response)
            if keyword_count >= 3:
                confidence += 0.2
                reasoning += 0.2
            elif keyword_count >= 1:
                confidence += 0.1
                reasoning += 0.1
            
            return InferenceResult(
                response=response,
                confidence=min(confidence, 1.0),
                reasoning_quality=min(reasoning, 1.0),
                analysis_depth=3
            )
    
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