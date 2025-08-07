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
    """데이터 처리 클래스 - 한국어 특화"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.answer_extraction_patterns = self._build_extraction_patterns()
        self.validation_rules = self._build_validation_rules()
        
        # 성능 캐시
        self.structure_cache = {}
        self.pattern_cache = {}
        
        # 컴파일된 정규식
        self.compiled_patterns = self._compile_all_patterns()
        
        # 통계적 학습 데이터
        self.answer_statistics = self._initialize_statistics()
        
        # 한국어 정리 패턴 - 확장
        self.korean_cleanup_patterns = self._build_comprehensive_korean_patterns()
        
    def _build_comprehensive_korean_patterns(self) -> Dict[str, str]:
        """포괄적 한국어 정리 패턴"""
        return {
            # 중국어 간체/번체 -> 한국어
            r'[軟软][件体體]': '소프트웨어',
            r'[硬硬][件体體]': '하드웨어',
            r'[危険險]害': '위험',
            r'可能性': '가능성',
            r'[存在]': '존재',
            r'程[式序]': '프로그램',
            r'金融': '금융',
            r'交易': '거래',
            r'安全': '안전',
            r'保[險险險]': '보험',
            r'方案': '방안',
            r'[資资]訊': '정보',
            r'[系係][統统]': '시스템',
            r'管理': '관리',
            r'[技技][術术]': '기술',
            r'[服服][務务]': '서비스',
            r'[機机][構构]': '기관',
            r'[規规]定': '규정',
            r'法律': '법률',
            r'[責责]任': '책임',
            r'保[護护]': '보호',
            r'[處处]理': '처리',
            r'收集': '수집',
            r'利用': '이용',
            r'提供': '제공',
            r'同意': '동의',
            r'[個个]人': '개인',
            r'情[報报]': '정보',
            r'[電电]子': '전자',
            r'[認认][證证]': '인증',
            r'加密': '암호화',
            r'[網网][路络絡]': '네트워크',
            r'[數数][據据]': '데이터',
            r'[檔档]案': '파일',
            r'[儲储]存': '저장',
            r'[備备]份': '백업',
            r'[復复][原元]': '복원',
            r'[權权]限': '권한',
            r'[訪访][問问]': '접근',
            r'[控控]制': '통제',
            r'[監监][督督]': '감독',
            r'[審审][計计]': '감사',
            r'[評评][估价]': '평가',
            r'[風风][險险]': '위험',
            r'[對对][策책]': '대책',
            r'[預预]防': '예방',
            r'[應应][對对]': '대응',
            r'[緊紧]急': '긴급',
            r'[災灾]害': '재해',
            r'[恢复復]': '복구',
            
            # 일본어 잔재 제거
            r'[あ-ん]': '',
            r'[ア-ン]': '',
            r'[一-龯]': '',
            
            # 영어 기술 용어 -> 한국어
            r'\bsoftware\b': '소프트웨어',
            r'\bhardware\b': '하드웨어',
            r'\bsystem\b': '시스템',
            r'\bnetwork\b': '네트워크',
            r'\bsecurity\b': '보안',
            r'\bdata\b': '데이터',
            r'\bserver\b': '서버',
            r'\bclient\b': '클라이언트',
            r'\bbackup\b': '백업',
            r'\bpassword\b': '비밀번호',
            r'\baccess\b': '접근',
            r'\bcontrol\b': '통제',
            r'\bmanagement\b': '관리',
            r'\bpolicy\b': '정책',
            r'\bprocedure\b': '절차',
            r'\bprocess\b': '프로세스',
            r'\bservice\b': '서비스',
            r'\bapplication\b': '애플리케이션',
            r'\bdatabase\b': '데이터베이스',
            r'\bencryption\b': '암호화',
            r'\bauthentication\b': '인증',
            r'\bauthorization\b': '권한부여',
            r'\bfirewall\b': '방화벽',
            r'\bvirus\b': '바이러스',
            r'\bmalware\b': '악성코드',
            r'\bransomware\b': '랜섬웨어',
            r'\bphishing\b': '피싱',
            r'\bhacking\b': '해킹',
            r'\bincident\b': '사고',
            r'\bresponse\b': '대응',
            r'\brecovery\b': '복구',
            r'\bfinancial\b': '금융',
            r'\btransaction\b': '거래',
            r'\bpayment\b': '결제',
            r'\btransfer\b': '이체',
            r'\baccount\b': '계정',
            r'\bbalance\b': '잔액',
            r'\bcredit\b': '신용',
            r'\bdebit\b': '직불',
            r'\binsurance\b': '보험',
            r'\brisk\b': '위험',
            r'\bcompliance\b': '준수',
            r'\baudit\b': '감사',
            r'\breport\b': '보고서',
            r'\banalysis\b': '분석',
            r'\bmonitoring\b': '모니터링',
            r'\bprevention\b': '예방',
            r'\bdetection\b': '탐지',
            r'\binvestigation\b': '조사',
            r'\bforensic\b': '포렌식'
        }
    
    def _clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리 - 강화 버전"""
        
        if not text:
            return ""
        
        # 1단계: 이상한 유니코드 문자 제거
        text = re.sub(r'[\u0000-\u001f\u007f-\u009f]', '', text)
        
        # 2단계: 한자/영어 -> 한국어 변환
        for pattern, replacement in self.korean_cleanup_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 3단계: 남은 한자 완전 제거
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        text = re.sub(r'[\u3400-\u4dbf]+', '', text)  # 확장 한자
        
        # 4단계: 일본어 문자 제거
        text = re.sub(r'[\u3040-\u309f]+', '', text)  # 히라가나
        text = re.sub(r'[\u30a0-\u30ff]+', '', text)  # 가타카나
        
        # 5단계: 특수문자 정리 (한글, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣0-9.,!?()·\-\n]', '', text)
        
        # 6단계: 남은 영어 단어 제거 (괄호 안은 유지)
        text = re.sub(r'\b[A-Za-z]+\b(?!\))', '', text)
        
        # 7단계: 중복 공백 및 구두점 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # 8단계: 불완전한 문장 보완
        text = text.strip()
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def _validate_korean_text(self, text: str, question_type: str) -> Tuple[bool, float]:
        """한국어 텍스트 검증"""
        
        # 객관식은 숫자만 확인
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return True, 1.0
            return False, 0.0
        
        # 주관식 검증
        if not text or len(text.strip()) < 20:
            return False, 0.0
        
        # 한자 확인
        if re.search(r'[\u4e00-\u9fff]', text):
            return False, 0.0
        
        # 한국어 비율 계산
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return False, 0.0
        
        korean_ratio = korean_chars / total_chars
        
        # 최소 70% 한국어 필요
        if korean_ratio < 0.7:
            return False, korean_ratio
        
        # 영어 비율 체크
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if english_ratio > 0.2:
            return False, korean_ratio * (1 - english_ratio)
        
        return True, korean_ratio
    
    def _build_extraction_patterns(self) -> Dict[str, List[str]]:
        """답변 추출 패턴"""
        patterns = {
            "explicit_answer": [
                r'정답[:\s]*([1-5])',
                r'답[:\s]*([1-5])',
                r'최종\s*답[:\s]*([1-5])',
                r'^([1-5])$',
                r'^([1-5])\s*$'
            ],
            "choice_reference": [
                r'([1-5])번',
                r'선택지\s*([1-5])',
                r'([1-5])가\s*정답'
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
        """검증 규칙"""
        rules = {
            "choice_range": lambda x: x.isdigit() and 1 <= int(x) <= 5,
            "length_appropriate": lambda x: 20 <= len(x) <= 1500,
            "not_empty": lambda x: x.strip() != "",
            "meaningful_content": lambda x: len(x.split()) >= 5 if not x.isdigit() else True,
            "korean_content": lambda x: bool(re.search(r'[가-힣]', x)),
            "no_chinese_chars": lambda x: not bool(re.search(r'[\u4e00-\u9fff]', x)),
            "minimal_english": lambda x: len(re.findall(r'[A-Za-z]', x)) < len(x) * 0.2,
            "professional_content": lambda x: any(term in x for term in ['법', '규정', '보안', '관리', '정책', '조치', '체계']) if len(x) > 50 else True
        }
        return rules
    
    def analyze_question_structure(self, question: str) -> Dict:
        """질문 구조 분석"""
        
        # 캐시 확인
        q_hash = hashlib.md5(question.encode()).hexdigest()[:12]
        if q_hash in self.structure_cache:
            return self.structure_cache[q_hash]
        
        lines = question.strip().split("\n")
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
        
        # 선택지 패턴
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
        
        # 분석
        structure["complexity_score"] = self._calculate_complexity_score(question)
        structure["domain_hints"] = self._extract_domain_hints(question)
        structure["structural_features"] = self._analyze_structural_features(question, choices)
        
        # 캐시 저장
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
            r"아닌\s*것"
        ]
        
        compiled_negative = re.compile("|".join(negative_patterns), re.IGNORECASE)
        return bool(compiled_negative.search(question_text))
    
    def _calculate_complexity_score(self, question: str) -> float:
        """복잡도 점수 계산"""
        score = 0.0
        
        # 길이 기반
        length = len(question)
        score += min(length / 2000, 0.3)
        
        # 구조 복잡도
        line_count = question.count('\n')
        score += min(line_count / 15, 0.2)
        
        # 법령 관련성
        law_terms = len(re.findall(r'법|조|항|규정|시행령|시행규칙', question))
        score += min(law_terms * 0.05, 0.2)
        
        # 전문 용어 밀도
        tech_terms = len(re.findall(r'암호화|인증|해시|PKI|SSL|접근제어|보안|시스템', question))
        score += min(tech_terms * 0.03, 0.15)
        
        # 숫자 및 기호 복잡도
        numbers = len(re.findall(r'\d+', question))
        symbols = len(re.findall(r'[%@#$&*()]', question))
        score += min((numbers + symbols) * 0.01, 0.15)
        
        return min(score, 1.0)
    
    def _extract_domain_hints(self, question: str) -> List[str]:
        """도메인 힌트 추출"""
        domain_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "개인정보처리", "동의", "수집", "이용"],
            "전자금융": ["전자금융", "전자적장치", "전자거래", "접근매체", "전자서명"],
            "정보보안": ["정보보안", "보안관리", "접근통제", "보안정책", "취약점"],
            "암호화": ["암호화", "복호화", "해시", "전자서명", "인증서", "키"],
            "사이버보안": ["해킹", "악성코드", "피싱", "스미싱", "파밍"],
            "법령": ["법", "규정", "조항", "시행령", "시행규칙"]
        }
        
        detected_domains = []
        question_lower = question.lower()
        
        for domain, keywords in domain_keywords.items():
            match_count = sum(1 for keyword in keywords if keyword in question_lower)
            if match_count >= 1:
                detected_domains.append((domain, match_count / len(keywords)))
        
        # 신뢰도 순으로 정렬
        detected_domains.sort(key=lambda x: x[1], reverse=True)
        return [domain for domain, confidence in detected_domains if confidence > 0.1]
    
    def _analyze_structural_features(self, question: str, choices: List[Dict]) -> Dict:
        """구조적 특징 분석"""
        features = {
            "avg_choice_length": 0,
            "choice_length_variance": 0,
            "has_code_snippets": False,
            "has_tables": False,
            "has_lists": False,
            "question_to_choice_ratio": 0
        }
        
        if choices:
            choice_lengths = [choice["length"] for choice in choices]
            features["avg_choice_length"] = np.mean(choice_lengths)
            features["choice_length_variance"] = np.var(choice_lengths)
            features["question_to_choice_ratio"] = len(question) / np.mean(choice_lengths)
        
        # 코드 스니펫 감지
        features["has_code_snippets"] = bool(re.search(r'```|<code>|{|}|\(\)|function|class', question))
        
        # 표 감지
        features["has_tables"] = bool(re.search(r'\|.*\||\t.*\t', question))
        
        # 리스트 감지
        features["has_lists"] = bool(re.search(r'^\s*[-*]\s+|^\s*\d+\.\s+', question, re.MULTILINE))
        
        return features
    
    def extract_mc_answer_fast(self, response: str) -> str:
        """빠른 객관식 답변 추출"""
        
        # 한국어 정리
        cleaned_response = self._clean_korean_text(response)
        
        # 단순 숫자 확인
        if re.match(r'^[1-5]$', cleaned_response.strip()):
            return cleaned_response.strip()
        
        # 패턴 매칭
        for category in ["explicit_answer", "choice_reference"]:
            patterns = self.compiled_patterns.get(category, [])
            for pattern in patterns:
                match = pattern.search(cleaned_response)
                if match:
                    answer = match.group(1)
                    if self.validation_rules["choice_range"](answer):
                        return answer
        
        # 숫자 검색
        numbers = re.findall(r'[1-5]', cleaned_response)
        if numbers:
            return numbers[0]
        
        # 기본값
        return "3"
    
    def extract_answer_intelligently(self, response: str, question: str) -> ProcessedAnswer:
        """지능형 답변 추출"""
        
        # 한국어 정리
        cleaned_response = self._clean_korean_text(response)
        question_structure = self.analyze_question_structure(question)
        
        if question_structure["question_type"] == "multiple_choice":
            return self._extract_mc_answer_optimized(cleaned_response, question_structure)
        else:
            return self._extract_subjective_answer_optimized(cleaned_response, question_structure)
    
    def _extract_mc_answer_optimized(self, response: str, question_structure: Dict) -> ProcessedAnswer:
        """최적화 객관식 답변 추출"""
        
        # 단순 숫자 확인
        if re.match(r'^[1-5]$', response.strip()):
            return ProcessedAnswer(
                final_answer=response.strip(),
                confidence=0.9,
                extraction_method="direct",
                validation_passed=True,
                korean_quality=1.0
            )
        
        # 패턴 매칭
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
        
        # 숫자 검색
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            return ProcessedAnswer(
                final_answer=numbers[0],
                confidence=0.6,
                extraction_method="number_search",
                validation_passed=True,
                korean_quality=1.0
            )
        
        # 통계적 폴백
        return ProcessedAnswer(
            final_answer="3",
            confidence=0.3,
            extraction_method="statistical_fallback",
            validation_passed=False,
            korean_quality=1.0
        )
    
    def _extract_subjective_answer_optimized(self, response: str, 
                                          question_structure: Dict) -> ProcessedAnswer:
        """최적화 주관식 답변 추출"""
        
        # 한국어 품질 검증
        is_valid, korean_quality = self._validate_korean_text(response, "subjective")
        
        if not is_valid:
            # 폴백 답변 생성
            fallback = self._generate_domain_specific_fallback(question_structure)
            return ProcessedAnswer(
                final_answer=fallback,
                confidence=0.5,
                extraction_method="fallback",
                validation_passed=True,
                korean_quality=0.8
            )
        
        # 길이 조정
        if len(response) < 50:
            fallback = self._generate_domain_specific_fallback(question_structure)
            return ProcessedAnswer(
                final_answer=fallback,
                confidence=0.5,
                extraction_method="length_fallback",
                validation_passed=True,
                korean_quality=0.8
            )
        elif len(response) > 800:
            response = response[:797] + "..."
        
        return ProcessedAnswer(
            final_answer=response.strip(),
            confidence=0.7,
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
        elif "암호화" in domain_hints:
            return "중요 정보는 안전한 암호 알고리즘을 사용하여 암호화해야 합니다. 대칭키와 공개키 암호화를 적절히 활용하고, 안전한 키 관리 체계를 구축하며, 전송 구간과 저장 시 모두 암호화를 적용해야 합니다."
        else:
            return "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고 지속적인 관리와 개선을 수행해야 합니다. 위험평가를 통해 취약점을 식별하고, 적절한 보호대책을 구현하며, 정기적인 점검을 통해 안전성을 확보해야 합니다."
    
    def post_process_answer(self, raw_response: str, question: str,
                          question_type: str) -> str:
        """통합 후처리 함수"""
        
        # 한국어 정리
        cleaned_response = self._clean_korean_text(raw_response)
        
        if question_type == "multiple_choice":
            # 빠른 추출
            return self.extract_mc_answer_fast(cleaned_response)
        else:
            # 주관식 처리
            processed = self.extract_answer_intelligently(cleaned_response, question)
            
            if self.validate_final_answer(processed, question, question_type):
                return processed.final_answer
            else:
                # 폴백
                structure = self.analyze_question_structure(question)
                return self._generate_domain_specific_fallback(structure)
    
    def validate_final_answer(self, processed_answer: ProcessedAnswer,
                            question: str, question_type: str) -> bool:
        """최종 답변 검증"""
        
        answer = processed_answer.final_answer
        
        # 기본 검증
        if not self.validation_rules["not_empty"](answer):
            return False
        
        if question_type == "multiple_choice":
            return self.validation_rules["choice_range"](answer)
        else:
            # 주관식 검증
            validations = [
                self.validation_rules["length_appropriate"](answer),
                self.validation_rules["meaningful_content"](answer),
                self.validation_rules["korean_content"](answer),
                self.validation_rules["no_chinese_chars"](answer),
                self.validation_rules["minimal_english"](answer),
                self.validation_rules["professional_content"](answer),
                processed_answer.korean_quality > 0.5
            ]
            
            # 70% 이상 통과하면 유효
            return sum(validations) / len(validations) >= 0.7
    
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