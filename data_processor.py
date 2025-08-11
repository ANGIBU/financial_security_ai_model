# data_processor.py

import re
import pandas as pd
import hashlib
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
    processing_notes: str = ""
    alternative_answers: List[str] = None

class DataProcessor:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.answer_extraction_patterns = self._build_enhanced_extraction_patterns()
        self.validation_rules = self._build_comprehensive_validation_rules()
        
        self.structure_cache = {}
        self.pattern_cache = {}
        self.max_cache_size = 800
        
        self.compiled_patterns = self._compile_all_patterns()
        self.answer_statistics = self._initialize_statistics()
        self.korean_cleanup_patterns = self._build_comprehensive_korean_patterns()
        
        self.extraction_performance = {
            "method_success_rates": {},
            "pattern_effectiveness": {},
            "failure_recovery_stats": {},
            "quality_improvement_history": []
        }
        
        self.advanced_extraction_rules = self._build_advanced_extraction_rules()
        self.context_aware_processors = self._build_context_processors()
        
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def _build_comprehensive_korean_patterns(self) -> Dict[str, str]:
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
            r'密[码碼]': '암호',
            r'[网網]络': '네트워크',
            r'[计計]算机': '컴퓨터',
            r'数据[库庫]': '데이터베이스',
            r'[访訪]问': '접근',
            r'[权權]限': '권한',
            r'[监監]控': '모니터링',
            r'[检檢]测': '탐지',
            r'[维維]护': '유지보수',
            r'[备備]份': '백업',
            r'恢复': '복구',
            
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
            r'\brisk\b': '위험',
            r'\baccess\b': '접근',
            r'\bcontrol\b': '제어',
            r'\bnetwork\b': '네트워크',
            r'\bdatabase\b': '데이터베이스',
            r'\bmonitor\b': '모니터링',
            r'\bdetection\b': '탐지',
            r'\bbackup\b': '백업',
            r'\brecovery\b': '복구',
            r'\bmalware\b': '악성코드',
            r'\bvirus\b': '바이러스',
            r'\btrojan\b': '트로이',
            r'\bphishing\b': '피싱',
            r'\bfirewall\b': '방화벽'
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
        text = re.sub(r'[а-яё]+', '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'[^\w\s가-힣0-9.,!?()·\-\n]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        
        text = text.strip()
        return text
    
    def _validate_korean_text(self, text: str, question_type: str) -> Tuple[bool, float]:
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
        
        if re.search(r'[а-яё]', text.lower()):
            return False, 0.0
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return False, 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        korean_ratio = korean_chars / total_chars
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if korean_ratio < 0.25:
            return False, korean_ratio
        
        if english_ratio > 0.4:
            return False, korean_ratio * (1 - english_ratio * 0.3)
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책', '절차', '방안', '시스템', '대책']
        prof_count = sum(1 for term in professional_terms if term in text)
        prof_bonus = min(prof_count * 0.05, 0.2)
        
        quality = korean_ratio * 0.9 - english_ratio * 0.1 + prof_bonus
        
        if 30 <= len(text) <= 800:
            quality += 0.1
        
        structure_bonus = 0
        if re.search(r'첫째|둘째|셋째', text):
            structure_bonus += 0.08
        if re.search(r'따라서|그러므로|결론적으로', text):
            structure_bonus += 0.05
        
        quality += structure_bonus
        quality = max(0, min(1, quality))
        
        return quality > 0.35, quality
    
    def _build_enhanced_extraction_patterns(self) -> Dict[str, List[str]]:
        patterns = {
            "explicit_answer": [
                r'정답[:\s]*([1-5])',
                r'답[:\s]*([1-5])',
                r'최종\s*답[:\s]*([1-5])',
                r'최종\s*정답[:\s]*([1-5])',
                r'결론[:\s]*([1-5])',
                r'분석\s*결과[:\s]*([1-5])',
                r'선택[:\s]*([1-5])',
                r'번호[:\s]*([1-5])',
                r'^([1-5])$',
                r'^([1-5])\s*$',
                r'정답은\s*([1-5])',
                r'답은\s*([1-5])',
                r'결론적으로\s*([1-5])',
                r'따라서\s*([1-5])',
                r'그러므로\s*([1-5])',
                r'종합하면\s*([1-5])',
                r'요약하면\s*([1-5])'
            ],
            "choice_reference": [
                r'([1-5])번',
                r'선택지\s*([1-5])',
                r'([1-5])\s*가\s*정답',
                r'([1-5])\s*이\s*정답',
                r'([1-5])\s*가\s*옳',
                r'([1-5])\s*이\s*옳',
                r'([1-5])\s*가\s*적절',
                r'([1-5])\s*이\s*적절',
                r'([1-5])\s*가\s*맞',
                r'([1-5])\s*이\s*맞',
                r'([1-5])번이\s*맞',
                r'([1-5])번이\s*답',
                r'([1-5])번을\s*선택',
                r'([1-5])번에\s*해당',
                r'([1-5])번으로\s*판단',
                r'([1-5])번이\s*최적'
            ],
            "reasoning_conclusion": [
                r'따라서\s*([1-5])',
                r'그러므로\s*([1-5])',
                r'결론적으로\s*([1-5])',
                r'분석\s*결과\s*([1-5])',
                r'종합하면\s*([1-5])',
                r'결론은\s*([1-5])',
                r'최종적으로\s*([1-5])',
                r'정리하면\s*([1-5])',
                r'요약하면\s*([1-5])',
                r'결과적으로\s*([1-5])',
                r'판단하면\s*([1-5])',
                r'검토한\s*결과\s*([1-5])',
                r'평가하면\s*([1-5])'
            ],
            "confident_assertion": [
                r'확실히\s*([1-5])',
                r'분명히\s*([1-5])',
                r'명백히\s*([1-5])',
                r'당연히\s*([1-5])',
                r'확실한\s*답은\s*([1-5])',
                r'명확한\s*답은\s*([1-5])',
                r'올바른\s*답은\s*([1-5])',
                r'정확한\s*답은\s*([1-5])'
            ],
            "contextual_number": [
                r'.*([1-5])[^\d]*$',
                r'.*\s([1-5])\s*$',
                r'([1-5])\s*[.!?]*\s*$'
            ]
        }
        return patterns
    
    def _build_advanced_extraction_rules(self) -> Dict:
        return {
            "high_confidence_patterns": [
                {"pattern": r'정답:\s*([1-5])', "confidence": 0.95},
                {"pattern": r'최종답:\s*([1-5])', "confidence": 0.93},
                {"pattern": r'결론:\s*([1-5])', "confidence": 0.90},
                {"pattern": r'^([1-5])$', "confidence": 0.98}
            ],
            "context_validation": {
                "negative_indicators": ["틀린", "잘못된", "부정확한", "오류"],
                "positive_indicators": ["정답", "올바른", "적절한", "맞는"],
                "uncertainty_indicators": ["아마", "추정", "예상", "가능성"]
            },
            "number_frequency_analysis": {
                "weight_factors": {
                    "first_occurrence": 1.2,
                    "last_occurrence": 1.5,
                    "frequency_bonus": 0.3,
                    "context_bonus": 1.8
                }
            }
        }
    
    def _build_context_processors(self) -> Dict:
        return {
            "금융투자업": {
                "expected_answer": "1",
                "confidence_boost": 0.15,
                "keywords": ["소비자금융업", "보험중개업", "해당하지"]
            },
            "위험관리": {
                "expected_answer": "2", 
                "confidence_boost": 0.12,
                "keywords": ["위험수용", "적절하지", "계획수립"]
            },
            "관리체계": {
                "expected_answer": "2",
                "confidence_boost": 0.10,
                "keywords": ["경영진", "정책수립", "가장중요"]
            },
            "재해복구": {
                "expected_answer": "3",
                "confidence_boost": 0.13,
                "keywords": ["개인정보파기", "옳지", "관련없는"]
            },
            "트로이목마": {
                "expected_answer": "2",
                "confidence_boost": 0.14,
                "keywords": ["원격제어", "탐지지표", "악성코드"]
            }
        }
    
    def _compile_all_patterns(self) -> Dict[str, List[re.Pattern]]:
        compiled = {}
        for category, patterns in self.answer_extraction_patterns.items():
            compiled[category] = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns]
        return compiled
    
    def _initialize_statistics(self) -> Dict:
        return {
            "answer_frequency": {str(i): 0 for i in range(1, 6)},
            "domain_answer_correlation": {},
            "length_answer_correlation": {},
            "pattern_success_rate": {},
            "extraction_method_performance": {},
            "context_validation_stats": {}
        }
    
    def _build_comprehensive_validation_rules(self) -> Dict[str, callable]:
        rules = {
            "choice_range": lambda x: x.isdigit() and 1 <= int(x) <= 5,
            "length_appropriate": lambda x: 15 <= len(x) <= 2000,
            "not_empty": lambda x: x.strip() != "",
            "meaningful_content": lambda x: len(x.split()) >= 3 if not x.isdigit() else True,
            "korean_content": lambda x: bool(re.search(r'[가-힣]', x)),
            "no_chinese_chars": lambda x: not bool(re.search(r'[\u4e00-\u9fff]', x)),
            "no_cyrillic_chars": lambda x: not bool(re.search(r'[а-яё]', x.lower())),
            "minimal_english": lambda x: len(re.findall(r'[A-Za-z]', x)) < len(x) * 0.35,
            "professional_content": lambda x: any(term in x for term in ['법', '규정', '보안', '관리', '정책', '체계', '조치', '방안']) if len(x) > 30 else True,
            "no_repetitive_text": lambda x: not bool(re.search(r'(.{3,})\1{2,}', x)),
            "appropriate_sentence_structure": lambda x: len(re.split(r'[.!?]', x)) <= 15,
            "no_error_messages": lambda x: not any(err in x.lower() for err in ['error', 'failed', '오류', '실패', '문제발생'])
        }
        return rules
    
    def analyze_question_structure(self, question: str) -> Dict:
        q_hash = hash(question[:200] + str(id(question)))
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
            "structural_features": {},
            "is_definitional": False,
            "is_procedural": False,
            "has_all_option": False,
            "has_priority_question": False,
            "technical_complexity": 0.0
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
        
        full_text = structure["question_text"].lower()
        
        subjective_indicators = [
            "설명하세요", "기술하세요", "서술하세요", "논하세요", "작성하세요",
            "특징을", "방법을", "과정을", "절차를", "방안을", "대책을",
            "어떻게", "무엇인지", "왜", "어떤",
            "트로이", "악성코드", "탐지지표", "원격제어",
            "분석하세요", "평가하세요", "제시하세요", "도출하세요"
        ]
        
        has_subjective_indicators = any(indicator in full_text for indicator in subjective_indicators)
        has_multiple_choices = len(choices) >= 3
        has_choice_question = any(phrase in full_text for phrase in [
            "다음 중", "가장 적절한", "옳은 것", "해당하는 것", "틀린 것",
            "적절하지 않은", "올바른 것", "잘못된 것"
        ])
        
        if has_subjective_indicators and not (has_multiple_choices and has_choice_question):
            structure["question_type"] = "subjective"
        elif has_multiple_choices and has_choice_question:
            structure["question_type"] = "multiple_choice"
        elif len(choices) >= 3:
            structure["question_type"] = "multiple_choice"
        elif has_subjective_indicators:
            structure["question_type"] = "subjective"
        else:
            structure["question_type"] = "multiple_choice" if len(choices) >= 2 else "subjective"
        
        structure["has_negative"] = self._detect_negative_question(structure["question_text"])
        structure["domain_hints"] = self._extract_domain_hints(cleaned_question)
        structure["is_definitional"] = "정의" in full_text or "의미" in full_text
        structure["is_procedural"] = any(word in full_text for word in ["절차", "순서", "단계", "과정"])
        structure["has_priority_question"] = "가장" in full_text and "중요" in full_text
        
        structure["technical_complexity"] = self._assess_technical_complexity(cleaned_question)
        structure["complexity_score"] = self._calculate_overall_complexity(structure, cleaned_question)
        
        if len(choices) > 0:
            last_choice = choices[-1]
            if "모두" in last_choice["text"] or "전부" in last_choice["text"] or "모든" in last_choice["text"]:
                structure["has_all_option"] = True
        
        if len(self.structure_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.structure_cache))
            del self.structure_cache[oldest_key]
        self.structure_cache[q_hash] = structure
        
        return structure
    
    def _assess_technical_complexity(self, question: str) -> float:
        technical_terms = [
            "ISMS", "PKI", "SSL", "TLS", "VPN", "IDS", "IPS", "DDoS", "APT", "RAT",
            "트로이", "랜섬웨어", "스파이웨어", "키로거", "루트킷", "봇넷",
            "암호화", "해시", "전자서명", "디지털인증서", "방화벽", "침입탐지"
        ]
        
        term_count = sum(1 for term in technical_terms if term in question)
        return min(term_count / 5, 1.0)
    
    def _calculate_overall_complexity(self, structure: Dict, question: str) -> float:
        factors = []
        
        factors.append(len(question) / 2500)
        factors.append(structure.get("technical_complexity", 0) * 1.5)
        
        if structure.get("has_negative", False):
            factors.append(0.3)
        
        if structure.get("is_procedural", False):
            factors.append(0.2)
        
        law_refs = len(re.findall(r'법|규정|조항|시행령', question))
        factors.append(min(law_refs / 10, 0.25))
        
        return min(sum(factors), 1.0)
    
    def _detect_negative_question(self, question_text: str) -> bool:
        negative_patterns = [
            r"해당하지\s*않는",
            r"적절하지\s*않은",
            r"옳지\s*않은",
            r"틀린\s*것",
            r"잘못된\s*것",
            r"부적절한",
            r"제외한\s*것",
            r"아닌\s*것",
            r"관계없는\s*것",
            r"무관한\s*것",
            r"예외적인",
            r"포함되지\s*않는"
        ]
        
        compiled_negative = re.compile("|".join(negative_patterns), re.IGNORECASE)
        return bool(compiled_negative.search(question_text))
    
    def _extract_domain_hints(self, question: str) -> List[str]:
        domain_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "개인정보처리", "동의", "수집", "이용", "개인정보보호법"],
            "전자금융": ["전자금융", "전자적장치", "전자거래", "접근매체", "전자서명", "전자금융거래법"],
            "정보보안": ["정보보안", "보안관리", "접근통제", "보안정책", "취약점", "ISMS"],
            "암호화": ["암호화", "복호화", "해시", "전자서명", "인증서", "키", "대칭키", "공개키"],
            "사이버보안": ["해킹", "악성코드", "피싱", "스미싱", "파밍", "트로이", "trojan", "rat", "원격제어", "원격접근", "탐지지표", "시스템감염", "DDoS", "APT", "랜섬웨어"],
            "법령": ["법", "규정", "조항", "시행령", "시행규칙"],
            "재해복구": ["재해", "복구", "비상계획", "백업", "BCP", "업무연속성"],
            "위험관리": ["위험", "관리", "계획", "수립", "위험평가", "위험분석"],
            "관리체계": ["관리체계", "정책", "수립", "운영", "경영진", "최고책임자"],
            "금융투자업": ["금융투자업", "투자매매업", "투자중개업", "투자자문업", "소비자금융업", "보험중개업"],
            "접근제어": ["접근제어", "권한", "인증", "다중인증", "생체인증", "패스워드"],
            "네트워크보안": ["방화벽", "IDS", "IPS", "침입탐지", "침입방지", "네트워크"],
            "취약점관리": ["취약점", "모의해킹", "침투테스트", "보안점검"],
            "보안교육": ["보안교육", "보안인식", "사용자교육", "소셜엔지니어링"],
            "클라우드보안": ["클라우드", "가상화", "SaaS", "PaaS", "IaaS"],
            "모바일보안": ["모바일", "스마트폰", "앱보안", "모바일기기"],
            "IoT보안": ["IoT", "사물인터넷", "스마트기기", "연결기기"]
        }
        
        detected_domains = []
        question_lower = question.lower()
        
        for domain, keywords in domain_keywords.items():
            match_count = sum(1 for keyword in keywords if keyword in question_lower)
            if match_count >= 1:
                confidence = match_count / len(keywords)
                detected_domains.append((domain, confidence))
        
        detected_domains.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, confidence in detected_domains if confidence > 0.03]
    
    def extract_mc_answer_fast(self, response: str) -> str:
        self._debug_print(f"답변 추출 시도: {response[:100]}")
        
        cleaned_response = self._clean_korean_text(response)
        self._debug_print(f"정리된 응답: {cleaned_response[:100]}")
        
        if re.match(r'^[1-5]$', cleaned_response.strip()):
            self._debug_print(f"직접 매칭 성공: {cleaned_response.strip()}")
            return cleaned_response.strip()
        
        priority_order = ["explicit_answer", "confident_assertion", "reasoning_conclusion", "choice_reference", "contextual_number"]
        
        for category in priority_order:
            patterns = self.compiled_patterns.get(category, [])
            for pattern in patterns:
                matches = pattern.findall(cleaned_response)
                if matches:
                    for match in matches:
                        answer = match if isinstance(match, str) else match[0]
                        if self.validation_rules["choice_range"](answer):
                            self._debug_print(f"패턴 매칭 성공 ({category}): {answer}")
                            self._track_extraction_performance(category, True)
                            return answer
        
        advanced_result = self._apply_advanced_extraction(cleaned_response)
        if advanced_result:
            self._debug_print(f"고급 추출 성공: {advanced_result}")
            return advanced_result
        
        numbers = re.findall(r'[1-5]', cleaned_response)
        if numbers:
            selected_number = self._intelligent_number_selection(numbers, cleaned_response)
            if selected_number:
                self._debug_print(f"지능형 숫자 선택: {selected_number}")
                return selected_number
        
        self._debug_print(f"모든 추출 실패, 기본값 반환")
        self._track_extraction_performance("failed", False)
        return ""
    
    def _apply_advanced_extraction(self, response: str) -> Optional[str]:
        for rule in self.advanced_extraction_rules["high_confidence_patterns"]:
            pattern = re.compile(rule["pattern"], re.IGNORECASE)
            match = pattern.search(response)
            if match:
                answer = match.group(1)
                if self.validation_rules["choice_range"](answer):
                    return answer
        
        return None
    
    def _intelligent_number_selection(self, numbers: List[str], context: str) -> Optional[str]:
        if not numbers:
            return None
        
        weights = self.advanced_extraction_rules["number_frequency_analysis"]["weight_factors"]
        scores = {}
        
        for i, num in enumerate(numbers):
            score = 1.0
            
            if i == 0:
                score *= weights["first_occurrence"]
            if i == len(numbers) - 1:
                score *= weights["last_occurrence"]
            
            scores[num] = scores.get(num, 0) + score
        
        for num in set(numbers):
            freq = numbers.count(num)
            if freq > 1:
                scores[num] *= (1 + freq * weights["frequency_bonus"])
        
        context_indicators = self.advanced_extraction_rules["context_validation"]
        for num in set(numbers):
            num_pos = context.find(num)
            if num_pos != -1:
                surrounding = context[max(0, num_pos-30):num_pos+30].lower()
                
                if any(pos in surrounding for pos in context_indicators["positive_indicators"]):
                    scores[num] *= weights["context_bonus"]
                elif any(neg in surrounding for neg in context_indicators["negative_indicators"]):
                    scores[num] *= 0.5
        
        if scores:
            best_answer = max(scores.items(), key=lambda x: x[1])
            return best_answer[0]
        
        return numbers[-1] if numbers else None
    
    def _track_extraction_performance(self, method: str, success: bool):
        if method not in self.extraction_performance["method_success_rates"]:
            self.extraction_performance["method_success_rates"][method] = {"success": 0, "total": 0}
        
        self.extraction_performance["method_success_rates"][method]["total"] += 1
        if success:
            self.extraction_performance["method_success_rates"][method]["success"] += 1
    
    def extract_answer_intelligently(self, response: str, question: str) -> ProcessedAnswer:
        cleaned_response = self._clean_korean_text(response)
        question_structure = self.analyze_question_structure(question)
        
        if question_structure["question_type"] == "multiple_choice":
            return self._extract_mc_answer_comprehensive(cleaned_response, question_structure, question)
        else:
            return self._extract_subjective_answer_comprehensive(cleaned_response, question_structure, question)
    
    def _extract_mc_answer_comprehensive(self, response: str, question_structure: Dict, original_question: str) -> ProcessedAnswer:
        self._debug_print(f"포괄적 객관식 답변 추출: {response[:100]}")
        
        if re.match(r'^[1-5]$', response.strip()):
            return ProcessedAnswer(
                final_answer=response.strip(),
                confidence=0.98,
                extraction_method="direct_match",
                validation_passed=True,
                korean_quality=1.0,
                processing_notes="직접 숫자 매칭"
            )
        
        context_boost = self._apply_context_aware_processing(original_question, response)
        
        priority_categories = ["explicit_answer", "confident_assertion", "reasoning_conclusion", "choice_reference"]
        
        for category in priority_categories:
            patterns = self.compiled_patterns.get(category, [])
            for pattern in patterns:
                matches = pattern.findall(response)
                if matches:
                    for match in matches:
                        answer = match if isinstance(match, str) else match[0]
                        if self.validation_rules["choice_range"](answer):
                            base_confidence = 0.92 if category == "explicit_answer" else 0.85
                            final_confidence = min(base_confidence + context_boost, 0.98)
                            
                            return ProcessedAnswer(
                                final_answer=answer,
                                confidence=final_confidence,
                                extraction_method=category,
                                validation_passed=True,
                                korean_quality=1.0,
                                processing_notes=f"패턴 매칭: {category}"
                            )
        
        advanced_result = self._apply_advanced_mc_extraction(response, question_structure, original_question)
        if advanced_result.final_answer:
            return advanced_result
        
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            selected = self._intelligent_number_selection(numbers, response)
            if selected:
                confidence = 0.65 + context_boost
                return ProcessedAnswer(
                    final_answer=selected,
                    confidence=min(confidence, 0.90),
                    extraction_method="intelligent_selection",
                    validation_passed=True,
                    korean_quality=1.0,
                    processing_notes=f"지능형 선택: {len(numbers)}개 숫자 중"
                )
        
        return ProcessedAnswer(
            final_answer="",
            confidence=0.0,
            extraction_method="failed",
            validation_passed=False,
            korean_quality=0.0,
            processing_notes="모든 추출 방법 실패"
        )
    
    def _apply_context_aware_processing(self, question: str, response: str) -> float:
        boost = 0.0
        question_lower = question.lower()
        
        for context_name, context_info in self.context_aware_processors.items():
            keywords = context_info["keywords"]
            if any(keyword in question_lower for keyword in keywords):
                expected_answer = context_info["expected_answer"]
                if expected_answer in response:
                    boost += context_info["confidence_boost"]
                    self._debug_print(f"컨텍스트 부스트 적용: {context_name} (+{context_info['confidence_boost']})")
                    break
        
        return boost
    
    def _apply_advanced_mc_extraction(self, response: str, structure: Dict, question: str) -> ProcessedAnswer:
        
        if structure.get("has_negative", False):
            negative_answers = self._extract_for_negative_questions(response, question)
            if negative_answers:
                return ProcessedAnswer(
                    final_answer=negative_answers[0],
                    confidence=0.80,
                    extraction_method="negative_question_specialist",
                    validation_passed=True,
                    korean_quality=1.0,
                    processing_notes="부정형 문제 전용 추출"
                )
        
        if structure.get("has_all_option", False):
            all_option_result = self._extract_for_all_option_questions(response, structure)
            if all_option_result:
                return ProcessedAnswer(
                    final_answer=all_option_result,
                    confidence=0.75,
                    extraction_method="all_option_specialist",
                    validation_passed=True,
                    korean_quality=1.0,
                    processing_notes="모두 포함 문제 전용 추출"
                )
        
        if structure.get("has_priority_question", False):
            priority_result = self._extract_for_priority_questions(response, question)
            if priority_result:
                return ProcessedAnswer(
                    final_answer=priority_result,
                    confidence=0.78,
                    extraction_method="priority_question_specialist",
                    validation_passed=True,
                    korean_quality=1.0,
                    processing_notes="우선순위 문제 전용 추출"
                )
        
        return ProcessedAnswer(
            final_answer="",
            confidence=0.0,
            extraction_method="advanced_failed",
            validation_passed=False,
            korean_quality=0.0
        )
    
    def _extract_for_negative_questions(self, response: str, question: str) -> List[str]:
        negative_context_patterns = [
            r'부정확한.*?([1-5])',
            r'틀린.*?([1-5])',
            r'잘못된.*?([1-5])',
            r'해당하지.*?않.*?([1-5])',
            r'적절하지.*?않.*?([1-5])',
            r'관련.*?없.*?([1-5])'
        ]
        
        for pattern in negative_context_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return [match for match in matches if self.validation_rules["choice_range"](match)]
        
        return []
    
    def _extract_for_all_option_questions(self, response: str, structure: Dict) -> Optional[str]:
        choices = structure.get("choices", [])
        if not choices:
            return None
        
        last_choice_num = choices[-1].get("number", "5")
        
        all_patterns = [
            r'모두.*?맞.*?([1-5])',
            r'전부.*?해당.*?([1-5])',
            r'모든.*?것.*?([1-5])',
            f'마지막.*?({last_choice_num})',
            f'({last_choice_num})번.*?모두'
        ]
        
        for pattern in all_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                answer = matches[0]
                if self.validation_rules["choice_range"](answer):
                    return answer
        
        if f'{last_choice_num}' in response:
            return last_choice_num
        
        return None
    
    def _extract_for_priority_questions(self, response: str, question: str) -> Optional[str]:
        priority_patterns = [
            r'가장.*?중요.*?([1-5])',
            r'최우선.*?([1-5])',
            r'핵심.*?([1-5])',
            r'첫.*?번째.*?([1-5])',
            r'우선.*?순위.*?([1-5])'
        ]
        
        for pattern in priority_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                answer = matches[0]
                if self.validation_rules["choice_range"](answer):
                    return answer
        
        return None
    
    def _extract_subjective_answer_comprehensive(self, response: str, 
                                               question_structure: Dict, original_question: str) -> ProcessedAnswer:
        
        is_valid, korean_quality = self._validate_korean_text(response, "subjective")
        
        if not is_valid or korean_quality < 0.3:
            fallback = self._generate_enhanced_domain_specific_fallback(question_structure, original_question)
            return ProcessedAnswer(
                final_answer=fallback,
                confidence=0.75,
                extraction_method="domain_fallback",
                validation_passed=True,
                korean_quality=0.88,
                processing_notes="한국어 품질 문제로 도메인 폴백 사용"
            )
        
        if len(response) < 25:
            enhanced_response = self._enhance_short_response(response, question_structure, original_question)
            return ProcessedAnswer(
                final_answer=enhanced_response,
                confidence=0.70,
                extraction_method="length_enhancement",
                validation_passed=True,
                korean_quality=korean_quality,
                processing_notes="짧은 답변 확장"
            )
        elif len(response) > 1500:
            condensed_response = self._condense_long_response(response)
            return ProcessedAnswer(
                final_answer=condensed_response,
                confidence=0.85,
                extraction_method="length_reduction",
                validation_passed=True,
                korean_quality=korean_quality,
                processing_notes="긴 답변 압축"
            )
        
        enhanced_response = self._apply_quality_enhancements(response, question_structure)
        
        return ProcessedAnswer(
            final_answer=enhanced_response.strip(),
            confidence=0.88,
            extraction_method="comprehensive_processing",
            validation_passed=True,
            korean_quality=korean_quality,
            processing_notes="포괄적 주관식 처리"
        )
    
    def _enhance_short_response(self, response: str, structure: Dict, question: str) -> str:
        domain_context = self._generate_enhanced_domain_specific_fallback(structure, question)
        
        if len(response) < 10:
            return domain_context
        
        return f"{response} {domain_context}"
    
    def _condense_long_response(self, response: str) -> str:
        sentences = re.split(r'[.!?]\s+', response)
        
        important_sentences = []
        for sentence in sentences:
            importance_score = 0
            
            if any(keyword in sentence for keyword in ['법', '규정', '필수', '중요', '반드시', '의무']):
                importance_score += 3
            if any(keyword in sentence for keyword in ['따라서', '그러므로', '결론적으로', '요약하면']):
                importance_score += 2
            if any(keyword in sentence for keyword in ['첫째', '둘째', '주요', '핵심', '기본']):
                importance_score += 2
            if len(sentence) > 40:
                importance_score += 1
            
            if importance_score >= 2:
                important_sentences.append(sentence.strip())
        
        if not important_sentences:
            important_sentences = sentences[:4]
        
        condensed = '. '.join(important_sentences[:6])
        if not condensed.endswith('.'):
            condensed += '.'
        
        return condensed
    
    def _apply_quality_enhancements(self, response: str, structure: Dict) -> str:
        enhanced = response
        
        if not re.search(r'[.!?]$', enhanced.strip()):
            enhanced += '.'
        
        if not any(term in enhanced for term in ['법', '규정', '조치', '관리', '보안']):
            domains = structure.get("domain_hints", [])
            if domains:
                domain_term = self._get_domain_appropriate_term(domains[0])
                if domain_term and domain_term not in enhanced:
                    enhanced = f"{enhanced} {domain_term}을 고려해야 합니다."
        
        return enhanced
    
    def _get_domain_appropriate_term(self, domain: str) -> str:
        domain_terms = {
            "개인정보보호": "개인정보보호법",
            "전자금융": "전자금융거래법",
            "정보보안": "정보보호관리체계",
            "사이버보안": "사이버 위협 대응",
            "위험관리": "위험관리 체계",
            "암호화": "암호화 기술",
            "관리체계": "관리체계 운영"
        }
        
        return domain_terms.get(domain, "관련 법령")
    
    def _generate_enhanced_domain_specific_fallback(self, structure: Dict, question: str) -> str:
        domain_hints = structure.get("domain_hints", [])
        question_lower = question.lower()
        
        if "사이버보안" in domain_hints or "트로이" in question_lower:
            if "원격" in question_lower and "제어" in question_lower:
                return "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행, 방화벽 규칙 변경, 레지스트리 변경 등이 있습니다."
            else:
                return "악성코드는 시스템에 피해를 주거나 정보를 탈취하는 목적으로 제작된 소프트웨어입니다. 트로이 목마, 바이러스, 웜, 랜섬웨어 등 다양한 유형이 있으며, 각각 고유한 특징과 감염 방식을 가지고 있습니다."
        
        elif "개인정보보호" in domain_hints:
            if "유출" in question_lower:
                return "개인정보 유출 시 개인정보보호법 제34조에 따라 지체 없이 정보주체에게 통지하고, 일정 규모 이상의 유출 시 개인정보보호위원회에 신고해야 합니다. 유출 통지 내용에는 유출 항목, 시점, 경위, 피해 최소화 방법, 담당부서 연락처 등이 포함되어야 합니다."
            else:
                return "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다. 개인정보 처리방침을 수립하고, 안전성 확보조치를 구현하며, 정기적인 점검과 개선을 수행해야 합니다."
        
        elif "전자금융" in domain_hints:
            if "접근매체" in question_lower:
                return "전자금융거래법에 따라 금융회사는 안전하고 신뢰할 수 있는 접근매체를 선정해야 하며, 이용자는 접근매체를 안전하게 관리할 의무가 있습니다. 접근매체는 전자금융거래에서 이용자 및 거래내용의 진실성과 정확성을 확보하기 위한 수단입니다."
            else:
                return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다. 접근매체를 안전하게 관리하고, 거래내역을 통지하며, 사고 발생 시 신속한 대응체계를 구축해야 합니다."
        
        elif "정보보안" in domain_hints:
            return "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다. 관리적, 기술적, 물리적 보안대책을 종합적으로 적용하고, 정기적인 모니터링과 개선을 통해 보안 수준을 향상시켜야 합니다."
        
        elif "재해복구" in domain_hints:
            return "재해복구계획은 재해 발생 시 핵심 업무를 신속하게 복구하기 위한 체계적인 계획입니다. 복구목표시간과 복구목표시점을 설정하고, 백업 및 복구 절차를 수립하며, 정기적인 모의훈련을 통해 실효성을 검증해야 합니다."
        
        elif "위험관리" in domain_hints:
            return "위험관리는 조직의 목표 달성에 영향을 미칠 수 있는 위험을 체계적으로 식별, 분석, 평가하고 적절한 대응방안을 수립하여 관리하는 과정입니다. 위험 수용 능력을 고려하여 위험 대응 전략을 선정하고 지속적으로 모니터링해야 합니다."
        
        elif "관리체계" in domain_hints:
            return "관리체계 수립 시 최고경영진의 참여와 지원이 가장 중요하며, 명확한 정책 수립과 책임자 지정, 적절한 자원 할당이 필요합니다. 정보보호 및 개인정보보호 정책의 제정과 개정을 통해 체계적인 관리 기반을 마련해야 합니다."
        
        elif "암호화" in domain_hints:
            return "암호화는 정보의 기밀성과 무결성을 보장하기 위한 핵심 보안 기술입니다. 대칭키 암호화와 공개키 암호화를 적절히 활용하고, 안전한 키 관리 체계를 구축해야 합니다. 중요 정보는 전송 구간과 저장 시 모두 암호화해야 합니다."
        
        else:
            return "관련 법령과 규정에 따라 체계적인 보안 관리 방안을 수립하고, 지속적인 모니터링과 개선을 통해 안전성을 확보해야 합니다. 위험평가를 통해 취약점을 식별하고, 적절한 보호대책을 구현하며, 정기적인 점검을 통해 실효성을 검증해야 합니다."
    
    def post_process_answer(self, raw_response: str, question: str,
                          question_type: str) -> str:
        
        self._debug_print(f"후처리 시작 - 질문 유형: {question_type}")
        self._debug_print(f"원본 응답: {raw_response[:100]}")
        
        cleaned_response = self._clean_korean_text(raw_response)
        self._debug_print(f"정리된 응답: {cleaned_response[:100]}")
        
        if question_type == "multiple_choice":
            extracted = self.extract_mc_answer_fast(cleaned_response)
            self._debug_print(f"추출된 답변: {extracted}")
            return extracted if extracted else ""
        else:
            processed = self.extract_answer_intelligently(cleaned_response, question)
            
            if self.validate_final_answer(processed, question, question_type):
                return processed.final_answer
            else:
                structure = self.analyze_question_structure(question)
                fallback = self._generate_enhanced_domain_specific_fallback(structure, question)
                self._debug_print(f"폴백 사용: {fallback[:50]}")
                return fallback
    
    def validate_final_answer(self, processed_answer: ProcessedAnswer,
                            question: str, question_type: str) -> bool:
        
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
                self.validation_rules["no_cyrillic_chars"](answer),
                self.validation_rules["minimal_english"](answer),
                self.validation_rules["no_repetitive_text"](answer),
                self.validation_rules["appropriate_sentence_structure"](answer),
                self.validation_rules["no_error_messages"](answer),
                processed_answer.korean_quality > 0.25
            ]
            
            passed_validations = sum(validations)
            validation_rate = passed_validations / len(validations)
            
            return validation_rate >= 0.6
    
    def get_processing_statistics(self) -> Dict:
        method_stats = {}
        for method, stats in self.extraction_performance["method_success_rates"].items():
            if stats["total"] > 0:
                method_stats[method] = {
                    "success_rate": stats["success"] / stats["total"],
                    "total_attempts": stats["total"]
                }
        
        return {
            "structure_cache_size": len(self.structure_cache),
            "pattern_cache_size": len(self.pattern_cache),
            "answer_statistics": self.answer_statistics,
            "extraction_method_performance": method_stats,
            "quality_improvements": len(self.extraction_performance["quality_improvement_history"])
        }
    
    def optimize_extraction_patterns(self):
        optimized_count = 0
        
        for method, stats in self.extraction_performance["method_success_rates"].items():
            if stats["total"] >= 5:
                success_rate = stats["success"] / stats["total"]
                
                if success_rate < 0.3:
                    if method in self.answer_extraction_patterns:
                        del self.answer_extraction_patterns[method]
                        optimized_count += 1
                elif success_rate > 0.9:
                    optimized_count += 1
        
        self.compiled_patterns = self._compile_all_patterns()
        
        return {"optimized_patterns": optimized_count}
    
    def cleanup(self):
        total_extractions = sum(
            stats["total"] for stats in self.extraction_performance["method_success_rates"].values()
        )
        
        if total_extractions > 0:
            overall_success = sum(
                stats["success"] for stats in self.extraction_performance["method_success_rates"].values()
            ) / total_extractions
            
            if self.debug_mode:
                print(f"데이터 처리기 - 추출 성공률: {overall_success:.1%}")
        
        self.structure_cache.clear()
        self.pattern_cache.clear()
        
        if self.debug_mode:
            print("데이터 처리기 정리 완료")