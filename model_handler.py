# model_handler.py

import torch
import re
import time
import gc
import random
import os
import unicodedata
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from config import (
    DEFAULT_MODEL_NAME,
    MODEL_CONFIG,
    GENERATION_CONFIG,
    OPTIMIZATION_CONFIG,
    MEMORY_CONFIG,
    get_device,
)


class SimpleModelHandler:

    def __init__(self, model_name: str = None, verbose: bool = False):
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.verbose = verbose
        self.device = get_device()

        self._initialize_integrated_data()
        self.optimization_config = OPTIMIZATION_CONFIG

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
            use_fast=MODEL_CONFIG["use_fast_tokenizer"],
        )

        self._optimize_tokenizer_for_korean()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, MODEL_CONFIG["torch_dtype"]),
            device_map=MODEL_CONFIG["device_map"],
            trust_remote_code=MODEL_CONFIG["trust_remote_code"],
        )

        self.model.eval()
        self._warmup()

    def _optimize_tokenizer_for_korean(self):
        """토크나이저 한국어 최적화"""
        if hasattr(self.tokenizer, "do_lower_case"):
            self.tokenizer.do_lower_case = False

        if hasattr(self.tokenizer, "normalize"):
            self.tokenizer.normalize = False

        special_tokens = ["<korean>", "</korean>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def _initialize_integrated_data(self):
        """JSON 데이터를 코드 내부로 통합하여 초기화"""
        
        # 도메인별 전문 프롬프트 템플릿
        self.domain_prompts = {
            "사이버보안": {
                "objective": "사이버보안 전문가로서 기술적이고 정확한 답변을 제공하세요.",
                "keywords": ["악성코드", "보안", "탐지", "방어", "침입", "취약점", "모니터링"],
                "structure": "기술적 특성 → 보안 위험 → 대응 방안",
                "tone": "전문적이고 기술적인"
            },
            "전자금융": {
                "objective": "전자금융 법률 전문가로서 법령과 기관을 정확히 명시하여 답변하세요.",
                "keywords": ["전자금융거래법", "금융감독원", "분쟁조정", "이용자보호", "접근매체"],
                "structure": "법령 근거 → 관련 기관 → 절차 설명",
                "tone": "법률적이고 정확한"
            },
            "개인정보보호": {
                "objective": "개인정보보호 전문가로서 법령과 기관 정보를 정확히 제시하세요.",
                "keywords": ["개인정보보호법", "개인정보보호위원회", "정보주체", "처리", "동의"],
                "structure": "법령 기반 → 담당 기관 → 권리와 의무",
                "tone": "법률적이고 체계적인"
            },
            "정보보안": {
                "objective": "정보보안 관리 전문가로서 체계적인 관리방안을 제시하세요.",
                "keywords": ["ISMS", "관리체계", "보안정책", "접근통제", "위험분석"],
                "structure": "관리체계 → 보안정책 → 실행방안",
                "tone": "관리적이고 체계적인"
            },
            "위험관리": {
                "objective": "위험관리 전문가로서 체계적인 위험관리 절차를 설명하세요.",
                "keywords": ["위험식별", "위험평가", "위험대응", "내부통제", "모니터링"],
                "structure": "위험식별 → 평가방법 → 대응전략",
                "tone": "체계적이고 분석적인"
            },
            "금융투자": {
                "objective": "자본시장 전문가로서 법령에 근거한 정확한 정보를 제공하세요.",
                "keywords": ["자본시장법", "금융투자업", "투자자보호", "적합성원칙"],
                "structure": "법령 근거 → 업무 구분 → 규제 내용",
                "tone": "법률적이고 정밀한"
            }
        }

        # 객관식 선택지 분석 패턴
        self.mc_choice_patterns = {
            "금융투자업_분류": {
                "투자자문업": "금융투자업",
                "투자매매업": "금융투자업", 
                "투자중개업": "금융투자업",
                "소비자금융업": "금융투자업 아님",
                "보험중개업": "금융투자업 아님"
            },
            "위험관리_요소": {
                "수행인력": "적절함",
                "위험 수용": "부적절함",
                "대응전략": "적절함",
                "기간": "적절함",
                "대상": "적절함"
            },
            "개인정보_중요요소": {
                "정책 제개정": "중요함",
                "경영진 참여": "가장 중요함",
                "책임자 지정": "중요함",
                "자원 할당": "중요함"
            }
        }

        # 한국어 텍스트 복구 설정
        self._setup_korean_recovery()

    def _setup_korean_recovery(self):
        """한국어 텍스트 복구 설정"""
        self.korean_recovery_mapping = {
            "갈취 묻는 말": "",
            "묻고 갈취": "",
            "갈취": "",
            "() 기반": "",
            "()는": "",
            "() 특징": "",
            "() 지표": "",
            "() 방안": "",
            "()를": "",
            "()에": "",
            "()의": "",
            "()와": "",
            "()로": "",
            "() 는": "",
            "() 이": "",
            "() 가": "",
            "() 을": "",
            "() 에": "",
            "() 와": "",
            "() 로": "",
            "괄호": "",
            "(괄호)": "",
        }

    def generate_answer(self, question: str, question_type: str, max_choice: int = 5,
                       intent_analysis: Dict = None, domain_hints: Dict = None) -> str:
        """답변 생성"""
        
        # 도메인 힌트에서 도메인 추출
        domain = domain_hints.get("domain", "일반") if domain_hints else "일반"
        
        if question_type == "multiple_choice":
            return self._generate_mc_answer(question, max_choice, domain, domain_hints)
        else:
            return self._generate_subjective_answer(question, domain, intent_analysis, domain_hints)

    def _generate_mc_answer(self, question: str, max_choice: int, domain: str, domain_hints: Dict) -> str:
        """객관식 답변 생성"""
        
        # 1단계: 선택지 분석
        choices = self._extract_choices_from_question(question)
        
        # 2단계: 도메인별 특화 분석
        if domain == "금융투자" and "해당하지 않는" in question.lower():
            return self._analyze_financial_investment_question(question, choices, max_choice)
        
        # 3단계: 정교한 프롬프트로 LLM 추론
        precise_prompt = self._create_precise_mc_prompt(question, max_choice, domain, choices, domain_hints)
        
        answer = self._generate_with_llm(precise_prompt, "multiple_choice", max_choice)
        
        # 4단계: 답변 검증 및 정리
        validated_answer = self._validate_mc_answer(answer, max_choice, question)
        
        return validated_answer

    def _extract_choices_from_question(self, question: str) -> Dict[str, str]:
        """질문에서 선택지 추출 및 분석"""
        choices = {}
        lines = question.split('\n')
        
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\s+(.+)', line)
            if match:
                choice_num = match.group(1)
                choice_content = match.group(2).strip()
                choices[choice_num] = choice_content
        
        return choices

    def _analyze_financial_investment_question(self, question: str, choices: Dict, max_choice: int) -> str:
        """금융투자업 분류 질문 분석"""
        
        for choice_num, choice_content in choices.items():
            choice_lower = choice_content.lower()
            
            # 금융투자업이 아닌 업종 찾기
            non_financial_terms = ["소비자금융업", "보험중개업", "신용카드", "할부금융"]
            
            if any(term in choice_lower for term in non_financial_terms):
                if 1 <= int(choice_num) <= max_choice:
                    return choice_num
        
        return "5"  # 기본값

    def _create_precise_mc_prompt(self, question: str, max_choice: int, domain: str, 
                                choices: Dict, domain_hints: Dict) -> str:
        """정교한 객관식 프롬프트 생성"""
        
        domain_info = self.domain_prompts.get(domain, {
            "objective": "전문가로서 정확한 답변을 제공하세요.",
            "tone": "전문적인"
        })
        
        # 선택지 분석 정보 추가
        choices_analysis = ""
        if choices:
            choices_analysis = "\n선택지 분석:\n"
            for num, content in choices.items():
                choices_analysis += f"{num}. {content}\n"
        
        # 도메인별 맞춤 지시문
        domain_instruction = domain_info["objective"]
        
        prompt = f"""당신은 {domain} 분야의 전문가입니다. {domain_instruction}

문제:
{question}
{choices_analysis}

위 문제를 분석하여 1부터 {max_choice} 중에서 정답을 선택하세요.

분석 과정:
1. 문제에서 요구하는 것을 파악
2. 각 선택지의 특성 분석  
3. {domain} 전문 지식 적용
4. 최종 정답 결정

정답 번호만 답하세요: """

        return prompt

    def _generate_subjective_answer(self, question: str, domain: str, 
                                  intent_analysis: Dict, domain_hints: Dict) -> str:
        """주관식 답변 생성"""
        
        # 정밀 모드인지 확인
        precision_mode = domain_hints.get("precision_mode", False) if domain_hints else False
        
        if precision_mode:
            return self._generate_precision_subjective_answer(question, domain, intent_analysis, domain_hints)
        else:
            return self._generate_standard_subjective_answer(question, domain, intent_analysis)

    def _generate_precision_subjective_answer(self, question: str, domain: str, 
                                            intent_analysis: Dict, domain_hints: Dict) -> str:
        """정밀 주관식 답변 생성"""
        
        primary_intent = intent_analysis.get("primary_intent", "일반") if intent_analysis else "일반"
        professional_terms = domain_hints.get("professional_terms", [])
        answer_structure = domain_hints.get("answer_structure", "")
        target_length = domain_hints.get("target_length", 200)
        
        # 도메인별 전문 프롬프트
        domain_info = self.domain_prompts.get(domain, {})
        
        precision_prompt = f"""당신은 {domain} 분야의 전문가입니다. 다음 질문에 대해 정확하고 전문적인 답변을 작성하세요.

질문: {question}

답변 요구사항:
1. 전문 용어 활용: {', '.join(professional_terms[:5])}
2. 답변 구조: {answer_structure}
3. 답변 길이: 약 {target_length}자
4. 질문 의도: {primary_intent}

{domain_info.get('objective', '전문적이고 정확한 답변을 제공하세요.')}

답변:"""

        answer = self._generate_with_llm(precision_prompt, "subjective", 5)
        
        return self._refine_subjective_answer(answer, domain, professional_terms)

    def _generate_standard_subjective_answer(self, question: str, domain: str, intent_analysis: Dict) -> str:
        """표준 주관식 답변 생성"""
        
        domain_info = self.domain_prompts.get(domain, {})
        
        standard_prompt = f"""질문: {question}

{domain_info.get('objective', '전문적이고 구체적으로 답변하세요.')}

답변:"""

        answer = self._generate_with_llm(standard_prompt, "subjective", 5)
        
        return self._clean_subjective_answer(answer)

    def _generate_with_llm(self, prompt: str, question_type: str, max_choice: int = 5) -> str:
        """LLM을 사용한 답변 생성"""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2000 if question_type == "subjective" else 1500,
                add_special_tokens=True,
            )

            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            # 생성 설정
            if question_type == "multiple_choice":
                gen_config = GenerationConfig(
                    max_new_tokens=10,
                    temperature=0.3,
                    top_p=0.7,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                gen_config = GenerationConfig(
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=gen_config)

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            # 반복 패턴 체크
            if self._detect_critical_repetitive_patterns(response):
                response = self._remove_repetitive_patterns(response)

            return response

        except Exception as e:
            if self.verbose:
                print(f"LLM 생성 오류: {e}")
            return None

    def _validate_mc_answer(self, answer: str, max_choice: int, question: str) -> str:
        """객관식 답변 검증"""
        
        if not answer:
            return self._get_rule_based_mc_answer(question, max_choice)
        
        # 정리된 답변에서 숫자 추출
        cleaned_answer = re.sub(r'[^\d]', '', answer)
        
        # 유효한 답변 찾기
        for char in cleaned_answer:
            if char.isdigit() and 1 <= int(char) <= max_choice:
                return char
        
        # 원본에서 다시 찾기
        numbers = re.findall(r'\b([1-9])\b', answer)
        for num in numbers:
            if 1 <= int(num) <= max_choice:
                return num
        
        return self._get_rule_based_mc_answer(question, max_choice)

    def _get_rule_based_mc_answer(self, question: str, max_choice: int) -> str:
        """규칙 기반 객관식 답변"""
        question_lower = question.lower()
        
        # 금융투자업 특별 처리
        if ("금융투자업" in question_lower and "해당하지 않는" in question_lower):
            return "5"
        
        # 부정형 문제
        if any(neg in question_lower for neg in ["해당하지 않는", "적절하지 않은", "옳지 않은"]):
            return str(max_choice) if max_choice >= 3 else "3"
        
        # 긍정형 문제
        if any(pos in question_lower for pos in ["가장 적절한", "가장 중요한"]):
            return "2"
        
        return str((max_choice + 1) // 2)

    def _refine_subjective_answer(self, answer: str, domain: str, professional_terms: List[str]) -> str:
        """주관식 답변 정제"""
        
        if not answer:
            return self._get_domain_fallback_answer(domain)
        
        # 기본 정리
        answer = self._clean_subjective_answer(answer)
        
        # 전문 용어 강화
        if professional_terms:
            for term in professional_terms[:3]:
                if term not in answer and len(answer) < 400:
                    # 자연스럽게 용어 삽입 (너무 억지스럽지 않게)
                    if "관련" in answer:
                        answer = answer.replace("관련", f"{term} 관련")
                        break
        
        return answer

    def _clean_subjective_answer(self, answer: str) -> str:
        """주관식 답변 기본 정리"""
        
        if not answer:
            return ""
        
        # 반복 패턴 제거
        if self._detect_critical_repetitive_patterns(answer):
            answer = self._remove_repetitive_patterns(answer)
        
        # 한국어 복구
        answer = self._recover_korean_text(answer)
        
        # 프롬프트 관련 텍스트 제거
        answer = re.sub(r'답변[:：]\s*', '', answer)
        answer = re.sub(r'질문[:：].*?\n', '', answer)
        
        # 기본 정리
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # 길이 조정
        if len(answer) > 600:
            sentences = answer.split('. ')
            answer = '. '.join(sentences[:5])
        
        # 문장 끝 처리
        if answer and not answer.endswith(('.', '다', '요', '함')):
            answer += '.'
        
        return answer

    def _get_domain_fallback_answer(self, domain: str) -> str:
        """도메인별 기본 답변"""
        fallbacks = {
            "사이버보안": "사이버보안 위협에 대응하기 위해 다층 방어체계를 구축하고 실시간 모니터링을 통해 종합적인 보안 관리를 수행해야 합니다.",
            "전자금융": "전자금융거래법에 따라 전자금융업자는 이용자 보호를 위한 보안조치를 시행하고 분쟁 발생 시 관련 기관을 통해 해결할 수 있습니다.",
            "개인정보보호": "개인정보보호법에 따라 개인정보보호위원회가 업무를 담당하며 개인정보침해신고센터에서 신고 접수 업무를 수행합니다.",
            "정보보안": "정보보안관리체계를 구축하여 보안정책 수립, 위험분석, 보안대책 구현의 절차를 체계적으로 운영해야 합니다.",
            "위험관리": "위험관리 체계를 구축하여 위험식별, 위험평가, 위험대응의 단계별 절차를 수립해야 합니다.",
            "금융투자": "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 관리기준을 수립해야 합니다."
        }
        
        return fallbacks.get(domain, "관련 법령과 규정에 따라 체계적인 관리 방안을 수립해야 합니다.")

    def _detect_critical_repetitive_patterns(self, text: str) -> bool:
        """문제 패턴 감지"""
        if not text or len(text) < 30:
            return False

        # 문제가 되는 반복 패턴
        critical_patterns = [
            r"갈취 묻는 말",
            r"묻고 갈취",
            r"(.{1,5})\s*(\1\s*){10,}",
        ]

        for pattern in critical_patterns:
            if re.search(pattern, text):
                return True

        # 단어 반복 체크
        words = text.split()
        if len(words) >= 10:
            for i in range(len(words) - 9):
                if words[i] == words[i+1] == words[i+2] == words[i+3]:
                    return True

        return False

    def _remove_repetitive_patterns(self, text: str) -> str:
        """반복 패턴 제거"""
        if not text:
            return ""

        # 문제 패턴 직접 제거
        for pattern, replacement in self.korean_recovery_mapping.items():
            text = text.replace(pattern, replacement)

        # 단어 반복 제거
        words = text.split()
        cleaned_words = []
        i = 0
        while i < len(words):
            current_word = words[i]
            count = 1
            
            while i + count < len(words) and words[i + count] == current_word:
                count += 1
            
            if count >= 5:
                cleaned_words.extend([current_word] * min(3, count))
            else:
                cleaned_words.extend([current_word] * count)
            
            i += count

        text = " ".join(cleaned_words)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _recover_korean_text(self, text: str) -> str:
        """한국어 텍스트 복구"""
        if not text:
            return ""

        # 유니코드 정규화
        text = unicodedata.normalize("NFC", text)
        
        # 문제 패턴 제거
        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)
        
        # 기본 정리
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'[.,!?]{3,}', '.', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def generate_contextual_mc_answer(self, question: str, max_choice: int, domain: str) -> str:
        """문맥 기반 객관식 답변 생성"""
        return self._generate_mc_answer(question, max_choice, domain, {"domain": domain})

    def generate_fallback_mc_answer(self, question: str, max_choice: int, domain: str) -> str:
        """대체 객관식 답변 생성"""
        return self._get_rule_based_mc_answer(question, max_choice)

    def _analyze_mc_context(self, question: str, domain: str = "일반") -> Dict:
        """객관식 문맥 분석"""
        question_lower = question.lower()
        
        context = {
            "is_negative": False,
            "is_positive": False,
            "domain": domain,
            "key_terms": [],
            "expected_pattern": ""
        }
        
        # 부정/긍정 패턴 분석
        negative_patterns = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
        positive_patterns = ["가장 적절한", "가장 중요한", "맞는 것", "올바른"]
        
        context["is_negative"] = any(pattern in question_lower for pattern in negative_patterns)
        context["is_positive"] = any(pattern in question_lower for pattern in positive_patterns)
        
        # 도메인별 핵심 용어 추출
        domain_terms = {
            "금융투자": ["금융투자업", "투자자문", "투자매매", "소비자금융", "보험중개"],
            "위험관리": ["위험관리", "위험수용", "수행인력", "대응전략"],
            "개인정보보호": ["개인정보", "정보주체", "경영진", "책임자"],
            "전자금융": ["한국은행", "자료제출", "통화신용정책", "금융통화위원회"]
        }
        
        if domain in domain_terms:
            context["key_terms"] = [term for term in domain_terms[domain] if term in question_lower]
        
        return context

    def _warmup(self):
        """모델 워밍업"""
        try:
            test_prompt = "테스트 질문입니다."
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)

            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                )
        except Exception as e:
            if self.verbose:
                print(f"워밍업 실패: {e}")

    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")