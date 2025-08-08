# data_processor.py
"""
데이터 처리 시스템
"""

import re
import pandas as pd
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from knowledge_base import FinancialSecurityKnowledgeBase

@dataclass
class ProcessedAnswer:
    """처리된 답변 결과"""
    final_answer: str
    confidence: float
    extraction_method: str
    validation_passed: bool
    korean_quality: float

class DataProcessor:
    """데이터 처리 클래스"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.answer_extraction_patterns = self._build_extraction_patterns()
        self.validation_rules = self._build_validation_rules()
        
        self.structure_cache = {}
        self.pattern_cache = {}
        self.max_cache_size = 300
        
        self.compiled_patterns = self._compile_all_patterns()
        self.answer_statistics = self._initialize_statistics()
        self.korean_cleanup_patterns = self._build_comprehensive_korean_patterns()
        
    def _build_comprehensive_korean_patterns(self) -> Dict[str, str]:
        """포괄적 한국어 정리 패턴"""
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
            r'[服服][務务]': '서비스',
            r'[規规]定': '규정',
            r'法律': '법률',
            r'[個个]人': '개인',
            r'情[報报]': '정보',
            r'[電电]子': '전자',
            r'[認认][證证]': '인증',
            r'加密': '암호화',
            
            r'\bsoftware\b': '소프트웨어',
            r'\bhardware\b': '하드웨어',
            r'\bsystem\b': '시스템',
            r'\bsecurity\b': '보안',
            r'\bdata\b': '데이터',
            r'\bmanagement\b': '관리',
            r'\bpolicy\b': '정책',
            r'\bencryption\b': '암호화',
            r'\bauthentication\b': '인증',
            r'\bfinancial\b': '금융',
            r'\btransaction\b': '거래',
            r'\brisk\b': '위험'
        }
    
    def _clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리 (완화)"""
        
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
        
        text = text.strip()
        return text
    
    def _validate_korean_text(self, text: str, question_type: str) -> Tuple[bool, float]:
        """한국어 텍스트 검증 (완화)"""
        
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return True, 1.0
            if re.search(r'[1-5]', text):
                return True, 0.8
            return False, 0.0
        
        if not text or len(text.strip()) < 15:
            return False, 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return False, 0.0
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return False, 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio < 0.3:
            return False, korean_ratio
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if english_ratio > 0.3:
            return False, korean_ratio * (1 - english_ratio * 0.5)
        
        return True, korean_ratio
    
    def _build_extraction_patterns(self) -> Dict[str, List[str]]:
        """답변 추출 패턴 (강화)"""
        patterns = {
            "explicit_answer": [
                r'정답[:\s]*([1-5])',
                r'답[:\s]*([1-5])',
                r'최종\s*답[:\s]*([1-5])',
                r'선택[:\s]*([1-5])',
                r'번호[:\s]*([1-5])',
                r'^([1-5])$',
                r'^([1-5])\s*$'
            ],
            "choice_reference": [
                r'([1-5])번',
                r'선택지\s*([1-5])',
                r'([1-5])\s*가\s*정답',
                r'([1-5])\s*이\s*정답',
                r'([1-5])\s*가\s*옳',
                r'([1-5])\s*이\s*옳',
                r'([1-5])\s*가\s*적절',
                r'([1-5])\s*이\s*적절'
            ],
            "reasoning_conclusion": [
                r'따라서\s*([1-5])',
                r'그러므로\s*([1-5])',
                r'결론적으로\s*([1-5])',
                r'분석\s*결과\s*([1-5])',
                r'종합하면\s*([1-5])',
                r'결론은\s*([1-5])'
            ]
        }
        return patterns
    
    def _compile_all_patterns(self) -> Dict[str, List[re.Pattern]]:
        """모든 패턴 컴파일"""
        compiled = {}
        for category, patterns in self.answer_extraction_patterns.items():
            compiled[category] = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
        return compiled
    
    def _initialize_statistics(self) -> Dict:
        """통계적 학습 데이터 초기화"""
        return {
            "answer_frequency": {str(i): 0 for i in range(1, 6)},
            "domain_answer_correlation": {},
            "length_answer_correlation": {},
            "pattern_success_rate": {}
        }
    
    def _build_validation_rules(self) -> Dict[str, callable]:
        """검증 규칙 (완화)"""
        rules = {
            "choice_range": lambda x: x.isdigit() and 1 <= int(x) <= 5,
            "length_appropriate": lambda x: 15 <= len(x) <= 1500,
            "not_empty": lambda x: x.strip() != "",
            "meaningful_content": lambda x: len(x.split()) >= 3 if not x.isdigit() else True,
            "korean_content": lambda x: bool(re.search(r'[가-힣]', x)),
            "no_chinese_chars": lambda x: not bool(re.search(r'[\u4e00-\u9fff]', x)),
            "minimal_english": lambda x: len(re.findall(r'[A-Za-z]', x)) < len(x) * 0.3,
            "professional_content": lambda x: any(term in x for term in ['법', '규정', '보안', '관리', '정책']) if len(x) > 30 else True
        }
        return rules
    
    def analyze_question_structure(self, question: str) -> Dict:
        """질문 구조 분석"""
        
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
            "structural_features": {}
        }
        
        question_parts = []
        choices = []
        
        choice_patterns = [
            re.compile(r"^\s*([1-5])\s+(.+)"),
            re.compile(r"^\s*([1-5])[.)]\s*(.+)"),
            re.compile(r"^\s*([①-⑤])\s*(.+)"),
            re.compile(r"^\s*\(?([1-5])\)?\s*(.+)"),
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
        structure["question_type"] = "multiple_choice" if len(choices) >= 2 else "subjective"
        structure["has_negative"] = self._detect_negative_question(structure["question_text"])
        structure["domain_hints"] = self._extract_domain_hints(cleaned_question)
        
        if len(self.structure_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.structure_cache))
            del self.structure_cache[oldest_key]
        self.structure_cache[q_hash] = structure
        
        return structure
    
    def _detect_negative_question(self, question_text: str) -> bool:
        """부정형 질문 감지"""
        negative_patterns = [
            r"해당하지\s*않는",
            r"적절하지\s*않은",
            r"옳지\s*않은",
            r"틀린\s*것",
            r"잘못된\s*것",
            r"부적절한",
            r"제외한\s*것",
            r"아닌\s*것",
            r"관계없는\s*것"
        ]
        
        compiled_negative = re.compile("|".join(negative_patterns), re.IGNORECASE)
        return bool(compiled_negative.search(question_text))
    
    def _extract_domain_hints(self, question: str) -> List[str]:
        """도메인 힌트 추출"""
        domain_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "개인정보처리", "동의", "수집", "이용"],
            "전자금융": ["전자금융", "전자적장치", "전자거래", "접근매체", "전자서명"],
            "정보보안": ["정보보안", "보안관리", "접근통제", "보안정책", "취약점"],
            "암호화": ["암호화", "복호화", "해시", "전자서명", "인증서", "키"],
            "사이버보안": ["해킹", "악성코드", "피싱", "스미싱", "파밍"],
            "법령": ["법", "규정", "조항", "시행령", "시행규칙"],
            "재해복구": ["재해", "복구", "비상계획", "백업"],
            "위험관리": ["위험", "관리", "계획", "수립"],
            "관리체계": ["관리체계", "정책", "수립", "운영"]
        }
        
        detected_domains = []
        question_lower = question.lower()
        
        for domain, keywords in domain_keywords.items():
            match_count = sum(1 for keyword in keywords if keyword in question_lower)
            if match_count >= 1:
                detected_domains.append((domain, match_count / len(keywords)))
        
        detected_domains.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, confidence in detected_domains if confidence > 0.1]
    
    def extract_mc_answer_fast(self, response: str) -> str:
        """빠른 객관식 답변 추출 (강화)"""
        
        print(f"[DEBUG] 답변 추출 시도: {response[:100]}")
        
        cleaned_response = self._clean_korean_text(response)
        print(f"[DEBUG] 정리된 응답: {cleaned_response[:100]}")
        
        if re.match(r'^[1-5]$', cleaned_response.strip()):
            print(f"[DEBUG] 직접 매칭 성공: {cleaned_response.strip()}")
            return cleaned_response.strip()
        
        for category in ["explicit_answer", "choice_reference", "reasoning_conclusion"]:
            patterns = self.compiled_patterns.get(category, [])
            for pattern in patterns:
                match = pattern.search(cleaned_response)
                if match:
                    answer = match.group(1)
                    if self.validation_rules["choice_range"](answer):
                        print(f"[DEBUG] 패턴 매칭 성공 ({category}): {answer}")
                        return answer
        
        numbers = re.findall(r'[1-5]', cleaned_response)
        if numbers:
            print(f"[DEBUG] 숫자 검색 성공: {numbers[0]}")
            return numbers[0]
        
        print(f"[DEBUG] 모든 추출 실패, 기본값 반환")
        return ""
    
    def extract_answer_intelligently(self, response: str, question: str) -> ProcessedAnswer:
        """지능형 답변 추출"""
        
        cleaned_response = self._clean_korean_text(response)
        question_structure = self.analyze_question_structure(question)
        
        if question_structure["question_type"] == "multiple_choice":
            return self._extract_mc_answer_optimized(cleaned_response, question_structure)
        else:
            return self._extract_subjective_answer_optimized(cleaned_response, question_structure)
    
    def _extract_mc_answer_optimized(self, response: str, question_structure: Dict) -> ProcessedAnswer:
        """최적화 객관식 답변 추출"""
        
        print(f"[DEBUG] 객관식 답변 추출: {response[:100]}")
        
        if re.match(r'^[1-5]$', response.strip()):
            return ProcessedAnswer(
                final_answer=response.strip(),
                confidence=0.9,
                extraction_method="direct",
                validation_passed=True,
                korean_quality=1.0
            )
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(response)
                if match:
                    answer = match.group(1)
                    if self.validation_rules["choice_range"](answer):
                        return ProcessedAnswer(
                            final_answer=answer,
                            confidence=0.8,
                            extraction_method=category,
                            validation_passed=True,
                            korean_quality=1.0
                        )
        
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            return ProcessedAnswer(
                final_answer=numbers[0],
                confidence=0.6,
                extraction_method="number_search",
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
    
    def _extract_subjective_answer_optimized(self, response: str, 
                                          question_structure: Dict) -> ProcessedAnswer:
        """최적화 주관식 답변 추출"""
        
        is_valid, korean_quality = self._validate_korean_text(response, "subjective")
        
        if not is_valid:
            fallback = self._generate_domain_specific_fallback(question_structure)
            return ProcessedAnswer(
                final_answer=fallback,
                confidence=0.6,
                extraction_method="fallback",
                validation_passed=True,
                korean_quality=0.8
            )
        
        if len(response) < 30:
            fallback = self._generate_domain_specific_fallback(question_structure)
            return ProcessedAnswer(
                final_answer=fallback,
                confidence=0.6,
                extraction_method="length_fallback",
                validation_passed=True,
                korean_quality=0.8
            )
        elif len(response) > 800:
            response = response[:797] + "..."
        
        return ProcessedAnswer(
            final_answer=response.strip(),
            confidence=0.8,
            extraction_method="subjective_processing",
            validation_passed=True,
            korean_quality=korean_quality
        )
    
    def _generate_domain_specific_fallback(self, structure: Dict) -> str:
        """도메인별 맞춤 폴백 생성"""
        domain_hints = structure.get("domain_hints", [])
        
        if "개인정보보호" in domain_hints:
            return "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다. 개인정보 처리방침을 수립하고, 안전성 확보조치를 구현하며, 정기적인 점검과 개선을 수행해야 합니다."
        elif "전자금융" in domain_hints:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다. 접근매체를 안전하게 관리하고, 거래내역을 통지하며, 사고 발생 시 신속한 대응체계를 구축해야 합니다."
        elif "정보보안" in domain_hints:
            return "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다. 관리적, 기술적, 물리적 보안대책을 종합적으로 적용하고, 정기적인 모니터링과 개선을 통해 보안 수준을 향상시켜야 합니다."
        elif "재해복구" in domain_hints:
            return "재해복구계획은 재해 발생 시 핵심 업무를 신속하게 복구하기 위한 체계적인 계획입니다. 복구목표시간과 복구목표시점을 설정하고, 백업 및 복구 절차를 수립하며, 정기적인 모의훈련을 통해 실효성을 검증해야 합니다."
        elif "위험관리" in domain_hints:
            return "위험관리는 조직의 목표 달성에 영향을 미칠 수 있는 위험을 체계적으로 식별, 분석, 평가하고 적절한 대응방안을 수립하여 관리하는 과정입니다. 위험 수용 능력을 고려하여 위험 대응 전략을 선정하고 지속적으로 모니터링해야 합니다."
        elif "관리체계" in domain_hints:
            return "관리체계 수립 시 최고경영진의 참여와 지원이 가장 중요하며, 명확한 정책 수립과 책임자 지정, 적절한 자원 할당이 필요합니다. 정보보호 및 개인정보보호 정책의 제정과 개정을 통해 체계적인 관리 기반을 마련해야 합니다."
        else:
            return "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고 지속적인 관리와 개선을 수행해야 합니다. 위험평가를 통해 취약점을 식별하고, 적절한 보호대책을 구현하며, 정기적인 점검을 통해 안전성을 확보해야 합니다."
    
    def post_process_answer(self, raw_response: str, question: str,
                          question_type: str) -> str:
        """통합 후처리 함수"""
        
        print(f"[DEBUG] 후처리 시작 - 질문 유형: {question_type}")
        print(f"[DEBUG] 원본 응답: {raw_response[:100]}")
        
        cleaned_response = self._clean_korean_text(raw_response)
        print(f"[DEBUG] 정리된 응답: {cleaned_response[:100]}")
        
        if question_type == "multiple_choice":
            extracted = self.extract_mc_answer_fast(cleaned_response)
            print(f"[DEBUG] 추출된 답변: {extracted}")
            return extracted if extracted else ""
        else:
            processed = self.extract_answer_intelligently(cleaned_response, question)
            
            if self.validate_final_answer(processed, question, question_type):
                return processed.final_answer
            else:
                structure = self.analyze_question_structure(question)
                fallback = self._generate_domain_specific_fallback(structure)
                print(f"[DEBUG] 폴백 사용: {fallback[:50]}")
                return fallback
    
    def validate_final_answer(self, processed_answer: ProcessedAnswer,
                            question: str, question_type: str) -> bool:
        """최종 답변 검증 (완화)"""
        
        answer = processed_answer.final_answer
        
        if not self.validation_rules["not_empty"](answer):
            return False
        
        if question_type == "multiple_choice":
            return self.validation_rules["choice_range"](answer)
        else:
            validations = [
                self.validation_rules["length_appropriate"](answer),
                self.validation_rules["meaningful_content"](answer),
                self.validation_rules["korean_content"](answer),
                self.validation_rules["no_chinese_chars"](answer),
                self.validation_rules["minimal_english"](answer),
                processed_answer.korean_quality > 0.3
            ]
            
            return sum(validations) / len(validations) >= 0.6
    
    def get_processing_statistics(self) -> Dict:
        """처리 통계 반환"""
        return {
            "structure_cache_size": len(self.structure_cache),
            "pattern_cache_size": len(self.pattern_cache),
            "answer_statistics": self.answer_statistics
        }
    
    def cleanup(self):
        """리소스 정리"""
        self.structure_cache.clear()
        self.pattern_cache.clear()
        print("데이터 처리기 정리 완료")