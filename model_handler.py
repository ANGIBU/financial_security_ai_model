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
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
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
        
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Torch Compile 활성화")
            except Exception:
                print("Torch Compile 실패, 기본 모델 사용")
        
        self._prepare_korean_tokens()
        
        self.response_cache = {}
        self.cache_hits = 0
        self.max_cache_size = 200
        
        print("모델 로딩 완료")
    
    def _prepare_korean_tokens(self):
        """한국어 토큰 준비"""
        self.korean_start_tokens = []
        korean_chars = "가나다라마바사아자차카타파하개내대래매배새애재채캐태패해"
        
        for char in korean_chars[:10]:
            tokens = self.tokenizer.encode(char, add_special_tokens=False)
            if tokens:
                self.korean_start_tokens.extend(tokens)
        
        self.korean_start_tokens = list(set(self.korean_start_tokens))
    
    def _create_korean_optimized_prompt(self, prompt: str, question_type: str) -> str:
        """한국어 최적화 프롬프트 생성 (더욱 강화)"""
        if question_type == "multiple_choice":
            korean_prefix = "### 매우 중요: 절대로 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요. 한자나 외국어 절대 금지 ###\n\n"
            korean_example = "\n### 올바른 답변 예시\n정답: 2\n\n### 절대 금지\n- 한자 사용 금지\n- 영어 사용 금지\n- 설명 금지\n\n"
        else:
            korean_prefix = "### 매우 중요: 반드시 순수 한국어로만 답변하세요. 한자, 영어, 일본어 등 모든 외국어 절대 금지 ###\n\n"
            korean_example = "\n### 올바른 답변 예시\n트로이 목마는 정상 프로그램으로 위장한 악성코드입니다.\n\n### 절대 금지\n- 한자 절대 금지\n- 영어 절대 금지\n- 일본어 절대 금지\n- 중국어 절대 금지\n\n"
        
        return korean_prefix + prompt + korean_example
    
    def generate_response(self, prompt: str, question_type: str,
                         max_attempts: int = 2) -> InferenceResult:
        """응답 생성"""
        
        start_time = time.time()
        
        cache_key = hash(prompt[:100])
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached = self.response_cache[cache_key]
            cached.inference_time = 0.01
            return cached
        
        optimized_prompt = self._create_korean_optimized_prompt(prompt, question_type)
        
        best_result = None
        best_score = 0
        
        for attempt in range(max_attempts):
            try:
                if question_type == "multiple_choice":
                    gen_config = GenerationConfig(
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        max_new_tokens=50,
                        repetition_penalty=1.1,
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
                        max_new_tokens=200,
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
                    max_length=1024
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
                
                print(f"[DEBUG] 시도 {attempt+1} 원본 응답: {raw_response[:100]}")
                
                cleaned_response = self._clean_korean_text(raw_response)
                print(f"[DEBUG] 정리된 응답: {cleaned_response[:100]}")
                
                if question_type == "multiple_choice":
                    extracted_answer = self._extract_mc_answer_enhanced(cleaned_response)
                    print(f"[DEBUG] 추출된 답변: {extracted_answer}")
                    
                    if extracted_answer and extracted_answer.isdigit() and 1 <= int(extracted_answer) <= 5:
                        result = InferenceResult(
                            response=extracted_answer,
                            confidence=0.8,
                            reasoning_quality=0.7,
                            analysis_depth=2,
                            korean_quality=1.0,
                            inference_time=time.time() - start_time
                        )
                        
                        if len(self.response_cache) >= self.max_cache_size:
                            oldest_key = next(iter(self.response_cache))
                            del self.response_cache[oldest_key]
                        self.response_cache[cache_key] = result
                        
                        print(f"[DEBUG] 성공적인 답변 추출: {extracted_answer}")
                        return result
                    else:
                        print(f"[DEBUG] 답변 추출 실패, 계속 시도...")
                        continue
                
                else:
                    korean_quality = self._evaluate_korean_quality_relaxed(cleaned_response, question_type)
                    print(f"[DEBUG] 한국어 품질: {korean_quality}")
                    
                    if korean_quality > 0.3:
                        result = self._evaluate_response(cleaned_response, question_type)
                        result.korean_quality = korean_quality
                        result.inference_time = time.time() - start_time
                        
                        score = korean_quality * result.confidence
                        if score > best_score:
                            best_score = score
                            best_result = result
                    
                    if korean_quality > 0.6:
                        break
                        
            except Exception as e:
                print(f"[DEBUG] 생성 오류 (시도 {attempt+1}): {e}")
                continue
        
        if best_result is None:
            print(f"[DEBUG] 모든 시도 실패, 폴백 생성")
            best_result = self._create_fallback_result(question_type)
            best_result.inference_time = time.time() - start_time
        
        return best_result
    
    def _extract_mc_answer_enhanced(self, text: str) -> str:
        """강화된 객관식 답변 추출"""
        
        patterns = [
            r'정답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'최종\s*답[:\s]*([1-5])',
            r'선택[:\s]*([1-5])',
            r'번호[:\s]*([1-5])',
            r'^([1-5])$',
            r'^([1-5])\s*$',
            r'([1-5])번',
            r'선택지\s*([1-5])',
            r'([1-5])\s*가\s*정답',
            r'([1-5])\s*이\s*정답',
            r'따라서\s*([1-5])',
            r'그러므로\s*([1-5])',
            r'결론적으로\s*([1-5])',
            r'분석\s*결과\s*([1-5])',
            r'종합하면\s*([1-5])'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                answer = matches[0]
                if answer.isdigit() and 1 <= int(answer) <= 5:
                    return answer
        
        numbers = re.findall(r'[1-5]', text)
        if numbers:
            return numbers[0]
        
        return ""
    
    def _clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리 (완화)"""
        
        if not text:
            return ""
        
        chinese_to_korean = {
            r'軟件|软件': '소프트웨어',
            r'金融': '금융',
            r'交易': '거래',
            r'安全': '안전',
            r'管理': '관리',
            r'個人|个人': '개인',
            r'資訊|资讯': '정보',
            r'電子|电子': '전자'
        }
        
        for pattern, replacement in chinese_to_korean.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        text = re.sub(r'[^\w\s가-힣0-9.,!?()·\-\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _evaluate_korean_quality_relaxed(self, text: str, question_type: str) -> float:
        """완화된 한국어 품질 평가"""
        
        if not text:
            return 0.0
        
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return 1.0
            if re.search(r'[1-5]', text):
                return 0.8
            return 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.1
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        quality = korean_ratio * 0.6
        quality -= english_ratio * 0.2
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality += min(prof_count * 0.1, 0.3)
        
        if len(text) > 30:
            quality += 0.1
        
        return max(0, min(1, quality))
    
    def _get_korean_fallback_response(self, prompt: str) -> str:
        """한국어 폴백 응답"""
        
        if "개인정보" in prompt:
            return "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다."
        elif "전자금융" in prompt:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호하기 위한 방안이 요구됩니다."
        elif "보안" in prompt or "암호" in prompt:
            return "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가 및 개선이 필요합니다."
        else:
            return "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고 지속적인 관리와 개선이 필요합니다."
    
    def _create_fallback_result(self, question_type: str) -> InferenceResult:
        """폴백 결과 생성"""
        
        if question_type == "multiple_choice":
            return InferenceResult(
                response="2",
                confidence=0.3,
                reasoning_quality=0.3,
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
            if 50 <= length <= 500:
                confidence += 0.2
            elif 30 <= length < 50:
                confidence += 0.1
            
            keywords = ['법', '규정', '보안', '관리', '조치', '정책', '체계']
            keyword_count = sum(1 for k in keywords if k in response)
            if keyword_count >= 2:
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