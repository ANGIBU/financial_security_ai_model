# model_handler.py

"""
모델 핸들러
- LLM 모델 로딩 및 관리
- 텍스트 생성 및 추론
- 한국어 최적화 설정
- 메모리 관리
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
    response: str
    confidence: float
    reasoning_quality: float
    analysis_depth: int
    inference_time: float = 0.0
    korean_quality: float = 0.0

class ModelHandler:
    
    def __init__(self, model_name: str, device: str = "cuda", 
                 load_in_4bit: bool = True, max_memory_gb: int = 22, verbose: bool = False):
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose
        
        if self.verbose:
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
            if self.verbose:
                print("Flash Attention 2 활성화")
        except ImportError:
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                model_kwargs["attn_implementation"] = "sdpa"
                if self.verbose:
                    print("SDPA Attention 활성화")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.eval()
        
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                if self.verbose:
                    print("Torch Compile 활성화")
            except Exception:
                if self.verbose:
                    print("Torch Compile 실패, 기본 모델 사용")
        
        self._prepare_advanced_korean_optimization()
        
        self.response_cache = {}
        self.cache_hits = 0
        self.max_cache_size = 800
        self.memory_cleanup_counter = 0
        
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "korean_quality_scores": [],
            "avg_inference_time": 0.0,
            "cache_efficiency": 0.0
        }
        
        if self.verbose:
            print("모델 로딩 완료")
    
    def _prepare_advanced_korean_optimization(self):
        self.korean_tokens = []
        korean_syllables = [
            '가', '나', '다', '라', '마', '바', '사', '아', '자', '차', '카', '타', '파', '하',
            '개', '내', '대', '래', '매', '배', '새', '애', '재', '채', '캐', '태', '패', '해',
            '거', '너', '더', '러', '머', '버', '서', '어', '저', '처', '커', '터', '퍼', '허',
            '게', '네', '데', '레', '메', '베', '세', '에', '제', '체', '케', '테', '페', '헤',
            '고', '노', '도', '로', '모', '보', '소', '오', '조', '초', '코', '토', '포', '호',
            '구', '누', '두', '루', '무', '부', '수', '우', '주', '추', '쿠', '투', '푸', '후',
            '그', '느', '드', '르', '므', '브', '스', '으', '즈', '츠', '크', '트', '프', '흐',
            '기', '니', '디', '리', '미', '비', '시', '이', '지', '치', '키', '티', '피', '히'
        ]
        
        for syllable in korean_syllables:
            tokens = self.tokenizer.encode(syllable, add_special_tokens=False)
            if tokens:
                self.korean_tokens.extend(tokens)
        
        korean_words = [
            '법률', '규정', '관리', '보안', '정보', '시스템', '조치', '방안', '절차', '정책',
            '개인정보', '전자금융', '위험관리', '암호화', '인증', '접근', '통제', '감사', '평가'
        ]
        
        for word in korean_words:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if tokens:
                self.korean_tokens.extend(tokens)
        
        self.korean_tokens = list(set(self.korean_tokens))
        
        self.bad_words_ids = []
        bad_patterns = [
            ["软", "件"], ["軟", "件"], ["硬", "件"], ["金", "融"], ["電", "子"], 
            ["個", "人"], ["資", "訊"], ["管", "理"], ["安", "全"], ["交", "易"],
            ["系", "統"], ["网", "络"], ["網", "絡"], ["数", "据"], ["數", "據"],
            ["认", "证"], ["認", "證"], ["加", "密"], ["解", "密"], ["备", "份"],
            ["備", "份"], ["恢", "复"], ["恢", "復"], ["监", "控"], ["監", "控"]
        ]
        
        for pattern in bad_patterns:
            try:
                tokens = self.tokenizer.encode("".join(pattern), add_special_tokens=False)
                if tokens:
                    self.bad_words_ids.append(tokens)
            except:
                continue
        
        english_tech_terms = [
            "SQL", "HTML", "CSS", "JavaScript", "Python", "Java", "API", "URL",
            "HTTP", "HTTPS", "TCP", "UDP", "IP", "DNS", "SSL", "TLS"
        ]
        
        for term in english_tech_terms:
            try:
                tokens = self.tokenizer.encode(term, add_special_tokens=False)
                if tokens and len(tokens) > 1:
                    self.bad_words_ids.append(tokens)
            except:
                continue
        
        self.korean_grammar_patterns = [
            (r'([가-힣]+)을\s+([가-힣]+)', self._check_object_marker),
            (r'([가-힣]+)는\s+([가-힣]+)', self._check_subject_marker),
            (r'([가-힣]+)에서\s+([가-힣]+)', self._check_location_marker),
            (r'([가-힣]+)으로\s+([가-힣]+)', self._check_direction_marker)
        ]
    
    def _check_object_marker(self, word: str) -> str:
        if not word:
            return word
        
        last_char = ord(word[-1])
        if 0xAC00 <= last_char <= 0xD7A3:
            jongseong = (last_char - 0xAC00) % 28
            if jongseong == 0:
                return word + "를"
            else:
                return word + "을"
        return word + "을"
    
    def _check_subject_marker(self, word: str) -> str:
        if not word:
            return word
        
        last_char = ord(word[-1])
        if 0xAC00 <= last_char <= 0xD7A3:
            jongseong = (last_char - 0xAC00) % 28
            if jongseong == 0:
                return word + "는"
            else:
                return word + "은"
        return word + "은"
    
    def _check_location_marker(self, word: str) -> str:
        return word + "에서"
    
    def _check_direction_marker(self, word: str) -> str:
        if not word:
            return word
        
        last_char = ord(word[-1])
        if 0xAC00 <= last_char <= 0xD7A3:
            jongseong = (last_char - 0xAC00) % 28
            if jongseong == 0:
                return word + "로"
            else:
                return word + "으로"
        return word + "으로"
    
    def _create_korean_optimized_prompt(self, prompt: str, question_type: str) -> str:
        if question_type == "multiple_choice":
            korean_prefix = "### 한국어 객관식 문제 ###\n반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요.\n외국어 사용 절대 금지.\n\n"
            korean_suffix = "\n\n### 답변 형식 ###\n정답: [숫자만]"
        else:
            korean_prefix = "### 한국어 전문가 답변 ###\n반드시 순수 한국어로만 답변하세요.\n한자, 영어, 기타 외국어 절대 금지.\n전문적이고 정확한 한국어 사용.\n\n"
            korean_suffix = "\n\n### 답변 규칙 ###\n- 순수 한국어만 사용\n- 80-300자 내외\n- 전문 용어의 정확한 한국어 표현\n\n답변:"
        
        return korean_prefix + prompt + korean_suffix
    
    def _apply_korean_grammar_correction(self, text: str) -> str:
        corrected = text
        
        corrections = [
            (r'해당되지', '해당하지'),
            (r'관련되어', '관련하여'),
            (r'포함되어', '포함하여'),
            (r'구성되어', '구성하여'),
            (r'\s+합니다\.', '합니다.'),
            (r'\s+입니다\.', '입니다.'),
            (r'([가-힣])을을', r'\1을'),
            (r'([가-힣])를를', r'\1를'),
            (r'([가-힣])은은', r'\1은'),
            (r'([가-힣])는는', r'\1는')
        ]
        
        for pattern, replacement in corrections:
            corrected = re.sub(pattern, replacement, corrected)
        
        return corrected
    
    def generate_response(self, prompt: str, question_type: str,
                         max_attempts: int = 3) -> InferenceResult:
        
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
                        temperature=0.25,
                        top_p=0.8,
                        top_k=20,
                        max_new_tokens=30,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True,
                        bad_words_ids=self.bad_words_ids,
                        no_repeat_ngram_size=2
                    )
                else:
                    gen_config = GenerationConfig(
                        do_sample=True,
                        temperature=0.4,
                        top_p=0.9,
                        top_k=50,
                        max_new_tokens=350,
                        repetition_penalty=1.02,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                        use_cache=True,
                        bad_words_ids=self.bad_words_ids
                    )
                
                inputs = self.tokenizer(
                    optimized_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1200,
                    padding=False
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
                
                cleaned_response = self._clean_korean_text_enhanced(raw_response)
                
                if question_type == "multiple_choice":
                    extracted_answer = self._extract_mc_answer_enhanced(cleaned_response)
                    
                    if extracted_answer and extracted_answer.isdigit() and 1 <= int(extracted_answer) <= 5:
                        result = InferenceResult(
                            response=extracted_answer,
                            confidence=0.92,
                            reasoning_quality=0.85,
                            analysis_depth=2,
                            korean_quality=1.0,
                            inference_time=time.time() - start_time
                        )
                        
                        self._manage_cache()
                        self.response_cache[cache_key] = result
                        self._update_generation_stats(result, True)
                        
                        return result
                    else:
                        continue
                
                else:
                    korean_quality = self._evaluate_korean_quality_enhanced(cleaned_response)
                    
                    if korean_quality > 0.4:
                        result = self._evaluate_response(cleaned_response, question_type)
                        result.korean_quality = korean_quality
                        result.inference_time = time.time() - start_time
                        
                        score = korean_quality * result.confidence
                        if score > best_score:
                            best_score = score
                            best_result = result
                    
                    if korean_quality > 0.7:
                        break
                        
            except Exception as e:
                if self.verbose:
                    print(f"생성 오류 (시도 {attempt+1}): {e}")
                continue
        
        if best_result is None:
            best_result = self._create_fallback_result(question_type)
            best_result.inference_time = time.time() - start_time
            self._update_generation_stats(best_result, False)
        else:
            self._update_generation_stats(best_result, True)
        
        self._perform_memory_cleanup()
        
        return best_result
    
    def _extract_mc_answer_enhanced(self, text: str) -> str:
        patterns = [
            r'정답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'^([1-5])$',
            r'^([1-5])\s*$',
            r'선택[:\s]*([1-5])',
            r'번호[:\s]*([1-5])',
            r'결론[:\s]*([1-5])'
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
    
    def _clean_korean_text_enhanced(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'[\u0000-\u001f\u007f-\u009f]', '', text)
        
        chinese_to_korean = {
            r'軟[体體]件|软件': '소프트웨어',
            r'硬[体體]件|硬件': '하드웨어',
            r'金融': '금융',
            r'交易': '거래',
            r'安全': '안전',
            r'管理': '관리',
            r'個人|个人': '개인',
            r'資[讯訊]|资讯': '정보',
            r'電子|电子': '전자',
            r'系[统統]|系统': '시스템',
            r'保[护護]|保护': '보호',
            r'認[证證]|认证': '인증',
            r'加密': '암호화',
            r'解密': '복호화',
            r'[网網][络絡]|网络': '네트워크',
            r'数据[库庫]|数据库': '데이터베이스',
            r'[访訪]问|访问': '접근',
            r'[权權]限|权限': '권한',
            r'[监監]控|监控': '모니터링',
            r'[检檢]测|检测': '탐지',
            r'[备備]份|备份': '백업',
            r'恢复': '복구',
            r'[设設][备備]|设备': '장비',
            r'[终終]端|终端': '단말기',
            r'[服務务][器噐]|服务器': '서버',
            r'[应應]用|应用': '응용프로그램'
        }
        
        for pattern, replacement in chinese_to_korean.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        text = re.sub(r'[\u3040-\u309f]+', '', text)
        text = re.sub(r'[\u30a0-\u30ff]+', '', text)
        text = re.sub(r'[а-яё]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\s가-힣0-9.,!?()·\-\n]', ' ', text)
        
        text = self._apply_korean_grammar_correction(text)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        text = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', text)
        
        text = re.sub(r'([가-힣])\s+([가-힣]{1,2})\s+([가-힣])', r'\1 \2\3', text)
        
        return text.strip()
    
    def _evaluate_korean_quality_enhanced(self, text: str) -> float:
        if not text:
            return 0.0
        
        quality_factors = {}
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.05
        
        if re.search(r'[а-яё]', text.lower()):
            return 0.05
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        quality_factors["korean_ratio"] = korean_ratio * 0.6
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        quality_factors["english_penalty"] = -english_ratio * 0.2
        
        professional_terms = [
            '법', '규정', '조치', '관리', '보안', '체계', '정책', '절차', 
            '방안', '시스템', '개인정보', '전자금융', '위험관리', '암호화'
        ]
        prof_count = sum(1 for term in professional_terms if term in text)
        quality_factors["professional_bonus"] = min(prof_count * 0.06, 0.25)
        
        if 40 <= len(text) <= 400:
            quality_factors["length_bonus"] = 0.15
        elif 20 <= len(text) < 40:
            quality_factors["length_bonus"] = 0.08
        else:
            quality_factors["length_bonus"] = 0
        
        structure_markers = ['첫째', '둘째', '따라서', '그러므로', '결론적으로', '또한', '그리고']
        structure_count = sum(1 for marker in structure_markers if marker in text)
        quality_factors["structure_bonus"] = min(structure_count * 0.05, 0.12)
        
        grammar_errors = len(re.findall(r'해당되지|관련되어|포함되어', text))
        quality_factors["grammar_penalty"] = -grammar_errors * 0.1
        
        repetition_penalty = 0
        words = text.split()
        if len(words) > 3:
            word_counts = {}
            for word in words:
                if len(word) > 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            for count in word_counts.values():
                if count > 2:
                    repetition_penalty += (count - 2) * 0.05
        
        quality_factors["repetition_penalty"] = -min(repetition_penalty, 0.2)
        
        final_quality = sum(quality_factors.values())
        
        return max(0, min(1, final_quality))
    
    def _create_fallback_result(self, question_type: str) -> InferenceResult:
        if question_type == "multiple_choice":
            return InferenceResult(
                response="2",
                confidence=0.5,
                reasoning_quality=0.4,
                analysis_depth=1,
                korean_quality=1.0
            )
        else:
            return InferenceResult(
                response="관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다.",
                confidence=0.7,
                reasoning_quality=0.6,
                analysis_depth=1,
                korean_quality=0.9
            )
    
    def _evaluate_response(self, response: str, question_type: str) -> InferenceResult:
        if question_type == "multiple_choice":
            confidence = 0.6
            reasoning = 0.6
            
            if re.match(r'^[1-5]$', response.strip()):
                confidence = 0.9
                reasoning = 0.8
            elif re.search(r'[1-5]', response):
                confidence = 0.75
                reasoning = 0.65
            
            return InferenceResult(
                response=response,
                confidence=confidence,
                reasoning_quality=reasoning,
                analysis_depth=2
            )
        
        else:
            confidence = 0.7
            reasoning = 0.7
            
            length = len(response)
            if 80 <= length <= 400:
                confidence += 0.15
            elif 50 <= length < 80:
                confidence += 0.08
            elif length > 400:
                confidence -= 0.1
            
            keywords = ['법', '규정', '보안', '관리', '조치', '정책', '체계', '방안', '절차']
            keyword_count = sum(1 for k in keywords if k in response)
            if keyword_count >= 4:
                confidence += 0.18
                reasoning += 0.18
            elif keyword_count >= 3:
                confidence += 0.12
                reasoning += 0.12
            elif keyword_count >= 2:
                confidence += 0.08
                reasoning += 0.08
            elif keyword_count >= 1:
                confidence += 0.04
                reasoning += 0.04
            
            structure_quality = 0
            if any(word in response for word in ['첫째', '둘째', '셋째']):
                structure_quality += 0.1
            if any(word in response for word in ['따라서', '그러므로', '결론적으로']):
                structure_quality += 0.08
            
            confidence += structure_quality
            reasoning += structure_quality
            
            return InferenceResult(
                response=response,
                confidence=min(confidence, 1.0),
                reasoning_quality=min(reasoning, 1.0),
                analysis_depth=3
            )
    
    def _manage_cache(self):
        if len(self.response_cache) >= self.max_cache_size:
            keys_to_remove = list(self.response_cache.keys())[:self.max_cache_size // 3]
            for key in keys_to_remove:
                del self.response_cache[key]
    
    def _perform_memory_cleanup(self):
        self.memory_cleanup_counter += 1
        
        if self.memory_cleanup_counter % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.memory_cleanup_counter % 50 == 0:
            if len(self.response_cache) > self.max_cache_size // 2:
                self._manage_cache()
    
    def _update_generation_stats(self, result: InferenceResult, success: bool):
        self.generation_stats["total_generations"] += 1
        if success:
            self.generation_stats["successful_generations"] += 1
        
        self.generation_stats["korean_quality_scores"].append(result.korean_quality)
        if len(self.generation_stats["korean_quality_scores"]) > 100:
            self.generation_stats["korean_quality_scores"] = self.generation_stats["korean_quality_scores"][-100:]
        
        total_time = self.generation_stats["avg_inference_time"] * (self.generation_stats["total_generations"] - 1)
        total_time += result.inference_time
        self.generation_stats["avg_inference_time"] = total_time / self.generation_stats["total_generations"]
        
        total_requests = self.generation_stats["total_generations"]
        self.generation_stats["cache_efficiency"] = self.cache_hits / max(total_requests, 1)
    
    def get_performance_stats(self) -> Dict:
        avg_korean_quality = 0.0
        if self.generation_stats["korean_quality_scores"]:
            avg_korean_quality = sum(self.generation_stats["korean_quality_scores"]) / len(self.generation_stats["korean_quality_scores"])
        
        success_rate = 0.0
        if self.generation_stats["total_generations"] > 0:
            success_rate = self.generation_stats["successful_generations"] / self.generation_stats["total_generations"]
        
        return {
            "model_name": self.model_name,
            "total_generations": self.generation_stats["total_generations"],
            "success_rate": success_rate,
            "avg_korean_quality": avg_korean_quality,
            "avg_inference_time": self.generation_stats["avg_inference_time"],
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache),
            "cache_efficiency": self.generation_stats["cache_efficiency"],
            "memory_cleanups": self.memory_cleanup_counter,
            "korean_tokens": len(self.korean_tokens),
            "bad_words_patterns": len(self.bad_words_ids)
        }
    
    def cleanup(self):
        if self.verbose:
            stats = self.get_performance_stats()
            print(f"모델 통계: 생성 성공률 {stats['success_rate']:.1%}, 한국어 품질 {stats['avg_korean_quality']:.2f}")
            print(f"캐시 효율성: {stats['cache_efficiency']:.1%}, 메모리 정리: {stats['memory_cleanups']}회")
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        self.response_cache.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.response_cache),
            "korean_optimization": {
                "korean_tokens": len(self.korean_tokens),
                "bad_words_patterns": len(self.bad_words_ids),
                "grammar_patterns": len(self.korean_grammar_patterns)
            }
        }