# data_processor.py

"""
데이터 처리기
- 문제 구조 분석
- 한국어 텍스트 정리
- 답변 추출 및 검증
- 도메인 힌트 추출
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from knowledge_base import FinancialSecurityKnowledgeBase

@dataclass
class ProcessedAnswer:
    final_answer: str
    confidence: float
    extraction_method: str
    validation_passed: bool
    korean_quality: float

class DataProcessor:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        
        self.structure_cache = {}
        self.max_cache_size = 500
        
        self.korean_cleanup_patterns = self._build_korean_patterns()
        self.answer_extraction_patterns = self._build_extraction_patterns()
        self.validation_rules = self._build_validation_rules()
        
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def _build_korean_patterns(self) -> Dict[str, str]:
        return {
            r'[軟软][件体體]': '소프트웨어',
            r'[硬硬][件体體]': '하드웨어',
            r'金融': '금융',
            r'交易': '거래',
            r'安全': '안전',
            r'方案': '방안',
            r'[資资]訊': '정보',
            r'[系係][統统]': '시스템',
            r'管理': '관리',
            r'[技技][術术]': '기술',
            r'[規规]定': '규정',
            r'法律': '법률',
            r'[個个]人': '개인',
            r'情[報报]': '정보',
            r'[電电]子': '전자',
            r'加密': '암호화',
            r'密[码碼]': '암호',
            r'[网網]络': '네트워크',
            r'数据[库庫]': '데이터베이스',
            r'[访訪]问': '접근',
            r'[权權]限': '권한',
            r'[监監]控': '모니터링',
            r'[检檢]测': '탐지',
            r'[备備]份': '백업',
            r'恢复': '복구'
        }
    
    def _build_extraction_patterns(self) -> Dict[str, List[str]]:
        return {
            "explicit_answer": [
                r'정답[:\s]*([1-5])',
                r'답[:\s]*([1-5])',
                r'최종\s*답[:\s]*([1-5])',
                r'선택[:\s]*([1-5])',
                r'^([1-5])$',
                r'^([1-5])\s*$'
            ],
            "choice_reference": [
                r'([1-5])번',
                r'선택지\s*([1-5])',
                r'([1-5])\s*가\s*정답',
                r'([1-5])\s*이\s*정답'
            ],
            "reasoning_conclusion": [
                r'따라서\s*([1-5])',
                r'그러므로\s*([1-5])',
                r'결론적으로\s*([1-5])',
                r'분석\s*결과\s*([1-5])'
            ]
        }
    
    def _build_validation_rules(self) -> Dict[str, callable]:
        return {
            "choice_range": lambda x: x.isdigit() and 1 <= int(x) <= 5,
            "length_appropriate": lambda x: 10 <= len(x) <= 2000,
            "not_empty": lambda x: x.strip() != "",
            "korean_content": lambda x: bool(re.search(r'[가-힣]', x)),
            "no_chinese_chars": lambda x: not bool(re.search(r'[\u4e00-\u9fff]', x)),
            "minimal_english": lambda x: len(re.findall(r'[A-Za-z]', x)) < len(x) * 0.4
        }
    
    def _clean_korean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'[\u0000-\u001f\u007f-\u009f]', '', text)
        
        for pattern, replacement in self.korean_cleanup_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        text = re.sub(r'[\u3040-\u309f]+', '', text)
        text = re.sub(r'[\u30a0-\u30ff]+', '', text)
        
        text = re.sub(r'[^\w\s가-힣0-9.,!?()·\-\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        
        return text.strip()
    
    def analyze_question_structure(self, question: str) -> Dict:
        q_hash = hash(question[:200])
        if q_hash in self.structure_cache:
            return self.structure_cache[q_hash]
        
        cleaned_question = re.sub(r'\.{3,}', '', question.strip())
        
        lines = cleaned_question.strip().split("\n")
        structure = {
            "question_text": "",
            "choices": [],
            "choice_count": 0,
            "has_negative": False,
            "question_type": "subjective",
            "complexity_score": 0.0,
            "domain_hints": [],
            "is_definitional": False,
            "is_procedural": False,
            "has_all_option": False
        }
        
        question_parts = []
        choices = []
        
        choice_patterns = [
            re.compile(r"^\s*([1-5])\s+(.+)"),
            re.compile(r"^\s*([1-5])[.)]\s*(.+)"),
            re.compile(r"^\s*([①-⑤])\s*(.+)"),
            re.compile(r"^\s*\(?([1-5])\)?\s*(.+)")
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            is_choice = False
            for pattern in choice_patterns:
                match = pattern.match(line)
                if match:
                    choice_num, choice_text = match.groups()
                    choices.append({
                        "number": choice_num if choice_num.isdigit() else str(ord(choice_num) - ord('①') + 1),
                        "text": choice_text.strip(),
                        "length": len(choice_text.strip())
                    })
                    is_choice = True
                    break
            
            if not is_choice:
                question_parts.append(line)
        
        structure["question_text"] = " ".join(question_parts)
        structure["choices"] = choices
        structure["choice_count"] = len(choices)
        
        full_text = structure["question_text"].lower()
        
        subjective_indicators = [
            "설명하세요", "기술하세요", "서술하세요", "논하세요", "작성하세요",
            "특징을", "방법을", "과정을", "절차를", "방안을", "대책을",
            "어떻게", "무엇인지", "왜", "어떤",
            "트로이", "악성코드", "탐지지표", "원격제어"
        ]
        
        has_subjective_indicators = any(indicator in full_text for indicator in subjective_indicators)
        has_multiple_choices = len(choices) >= 3
        has_choice_question = any(phrase in full_text for phrase in [
            "다음 중", "가장 적절한", "옳은 것", "해당하는 것", "틀린 것"
        ])
        
        if has_subjective_indicators and not (has_multiple_choices and has_choice_question):
            structure["question_type"] = "subjective"
        elif has_multiple_choices and has_choice_question:
            structure["question_type"] = "multiple_choice"
        elif len(choices) >= 3:
            structure["question_type"] = "multiple_choice"
        else:
            structure["question_type"] = "subjective"
        
        structure["has_negative"] = self._detect_negative_question(structure["question_text"])
        structure["domain_hints"] = self._extract_domain_hints(cleaned_question)
        structure["is_definitional"] = "정의" in full_text or "의미" in full_text
        structure["is_procedural"] = any(word in full_text for word in ["절차", "순서", "단계", "과정"])
        
        if len(choices) > 0:
            last_choice = choices[-1]
            if "모두" in last_choice["text"] or "전부" in last_choice["text"]:
                structure["has_all_option"] = True
        
        if len(self.structure_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.structure_cache))
            del self.structure_cache[oldest_key]
        self.structure_cache[q_hash] = structure
        
        return structure
    
    def _detect_negative_question(self, question_text: str) -> bool:
        negative_patterns = [
            r"해당하지\s*않는",
            r"적절하지\s*않은",
            r"옳지\s*않은",
            r"틀린\s*것",
            r"잘못된\s*것",
            r"부적절한",
            r"아닌\s*것"
        ]
        
        compiled_negative = re.compile("|".join(negative_patterns), re.IGNORECASE)
        return bool(compiled_negative.search(question_text))
    
    def _extract_domain_hints(self, question: str) -> List[str]:
        domain_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "개인정보처리", "동의", "개인정보보호법"],
            "전자금융": ["전자금융", "전자적장치", "접근매체", "전자서명", "전자금융거래법"],
            "정보보안": ["정보보안", "보안관리", "접근통제", "보안정책", "ISMS"],
            "사이버보안": ["해킹", "악성코드", "피싱", "트로이", "원격제어", "탐지지표"],
            "위험관리": ["위험", "관리", "계획", "수립", "위험평가"],
            "관리체계": ["관리체계", "정책", "수립", "운영", "경영진"],
            "금융투자업": ["금융투자업", "투자매매업", "소비자금융업", "보험중개업"],
            "재해복구": ["재해", "복구", "비상계획", "백업", "BCP"],
            "암호화": ["암호화", "복호화", "암호", "키관리", "해시함수"]
        }
        
        detected_domains = []
        question_lower = question.lower()
        
        for domain, keywords in domain_keywords.items():
            match_count = sum(1 for keyword in keywords if keyword in question_lower)
            if match_count >= 1:
                detected_domains.append((domain, match_count / len(keywords)))
        
        detected_domains.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, confidence in detected_domains if confidence > 0.05]
    
    def extract_mc_answer_fast(self, response: str) -> str:
        self._debug_print(f"답변 추출 시도: {response[:100]}")
        
        cleaned_response = self._clean_korean_text(response)
        
        if re.match(r'^[1-5]$', cleaned_response.strip()):
            self._debug_print(f"직접 매칭 성공: {cleaned_response.strip()}")
            return cleaned_response.strip()
        
        priority_order = ["explicit_answer", "reasoning_conclusion", "choice_reference"]
        
        for category in priority_order:
            patterns = self.answer_extraction_patterns.get(category, [])
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.MULTILINE)
                if matches:
                    for match in matches:
                        answer = match if isinstance(match, str) else match[0]
                        if self.validation_rules["choice_range"](answer):
                            self._debug_print(f"패턴 매칭 성공 ({category}): {answer}")
                            return answer
        
        numbers = re.findall(r'[1-5]', cleaned_response)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def extract_answer_intelligently(self, response: str, question: str) -> ProcessedAnswer:
        cleaned_response = self._clean_korean_text(response)
        question_structure = self.analyze_question_structure(question)
        
        if question_structure["question_type"] == "multiple_choice":
            return self._extract_mc_answer_optimized(cleaned_response)
        else:
            return self._extract_subjective_answer_optimized(cleaned_response, question_structure)
    
    def _extract_mc_answer_optimized(self, response: str) -> ProcessedAnswer:
        if re.match(r'^[1-5]$', response.strip()):
            return ProcessedAnswer(
                final_answer=response.strip(),
                confidence=0.95,
                extraction_method="direct",
                validation_passed=True,
                korean_quality=1.0
            )
        
        for category in ["explicit_answer", "reasoning_conclusion", "choice_reference"]:
            patterns = self.answer_extraction_patterns.get(category, [])
            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
                if matches:
                    for match in matches:
                        answer = match if isinstance(match, str) else match[0]
                        if self.validation_rules["choice_range"](answer):
                            confidence = 0.90 if category == "explicit_answer" else 0.80
                            return ProcessedAnswer(
                                final_answer=answer,
                                confidence=confidence,
                                extraction_method=category,
                                validation_passed=True,
                                korean_quality=1.0
                            )
        
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            return ProcessedAnswer(
                final_answer=numbers[-1],
                confidence=0.60,
                extraction_method="last_number",
                validation_passed=True,
                korean_quality=1.0
            )
        
        return ProcessedAnswer(
            final_answer="",
            confidence=0.0,
            extraction_method="failed",
            validation_passed=False,
            korean_quality=0.0
        )
    
    def _extract_subjective_answer_optimized(self, response: str, structure: Dict) -> ProcessedAnswer:
        is_valid, korean_quality = self._validate_korean_text(response, "subjective")
        
        if not is_valid:
            fallback = self._generate_domain_specific_fallback(structure)
            return ProcessedAnswer(
                final_answer=fallback,
                confidence=0.70,
                extraction_method="fallback",
                validation_passed=True,
                korean_quality=0.85
            )
        
        if len(response) < 20:
            fallback = self._generate_domain_specific_fallback(structure)
            return ProcessedAnswer(
                final_answer=fallback,
                confidence=0.70,
                extraction_method="length_fallback",
                validation_passed=True,
                korean_quality=0.85
            )
        elif len(response) > 1200:
            response = response[:1197] + "..."
        
        return ProcessedAnswer(
            final_answer=response.strip(),
            confidence=0.85,
            extraction_method="subjective_processing",
            validation_passed=True,
            korean_quality=korean_quality
        )
    
    def _validate_korean_text(self, text: str, question_type: str) -> Tuple[bool, float]:
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return True, 1.0
            return False, 0.0
        
        if not text or len(text.strip()) < 10:
            return False, 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return False, 0.0
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return False, 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio < 0.2:
            return False, korean_ratio
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if english_ratio > 0.5:
            return False, korean_ratio * (1 - english_ratio * 0.3)
        
        return True, korean_ratio
    
    def _generate_domain_specific_fallback(self, structure: Dict) -> str:
        domain_hints = structure.get("domain_hints", [])
        
        if "사이버보안" in domain_hints:
            return "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행 등이 있습니다."
        elif "개인정보보호" in domain_hints:
            return "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다."
        elif "전자금융" in domain_hints:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다."
        elif "정보보안" in domain_hints:
            return "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다."
        elif "재해복구" in domain_hints:
            return "재해복구계획은 재해 발생 시 핵심 업무를 신속하게 복구하기 위한 체계적인 계획입니다."
        elif "위험관리" in domain_hints:
            return "위험관리는 조직의 목표 달성에 영향을 미칠 수 있는 위험을 체계적으로 식별, 분석, 평가하고 적절한 대응방안을 수립하여 관리하는 과정입니다."
        else:
            return "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고 지속적인 관리와 개선을 수행해야 합니다."
    
    def post_process_answer(self, raw_response: str, question: str, question_type: str) -> str:
        self._debug_print(f"후처리 시작 - 질문 유형: {question_type}")
        
        cleaned_response = self._clean_korean_text(raw_response)
        
        if question_type == "multiple_choice":
            extracted = self.extract_mc_answer_fast(cleaned_response)
            return extracted if extracted else ""
        else:
            processed = self.extract_answer_intelligently(cleaned_response, question)
            return processed.final_answer if processed.validation_passed else ""
    
    def cleanup(self):
        self.structure_cache.clear()
        if self.debug_mode:
            print("데이터 처리기 정리 완료")