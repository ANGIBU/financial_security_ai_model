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

class DataProcessor:
    """데이터 처리 클래스"""
    
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
        
        # 한국어 정리 패턴
        self.korean_cleanup_patterns = self._build_korean_cleanup_patterns()
        
    def _build_korean_cleanup_patterns(self) -> Dict[str, str]:
        """한국어 정리 패턴"""
        return {
            # 한자 -> 한국어 변환
            r'[軟软][件体]': '소프트웨어',
            r'[危険]害': '위험',
            r'可能性': '가능성', 
            r'[存在]': '존재',
            r'程[式序]': '프로그램',
            r'金融': '금융',
            r'交易': '거래',
            r'安全': '안전',
            r'保險': '보험',
            r'方案': '방안',
            r'資訊': '정보',
            r'系統': '시스템',
            r'管理': '관리',
            r'技術': '기술',
            r'服務': '서비스',
            r'機構': '기관',
            r'規定': '규정',
            r'法律': '법률',
            r'責任': '책임',
            r'保護': '보호',
            r'處理': '처리',
            r'收集': '수집',
            r'利用': '이용',
            r'提供': '제공',
            r'同意': '동의',
            r'個人': '개인',
            r'情報': '정보',
            r'電子': '전자',
            r'認證': '인증',
            r'加密': '암호화',
            r'網路': '네트워크',
            
            # 영어 -> 한국어 변환 (일반적인 경우)
            r'\bfinancial\b': '금융',
            r'\btransaction\b': '거래', 
            r'\bsafety\b': '안전',
            r'\bsecurity\b': '보안',
            r'\binsurance\b': '보험',
            r'\bmethod\b': '방법',
            r'\bsystem\b': '시스템',
            r'\binformation\b': '정보',
            r'\bmanagement\b': '관리',
            r'\bservice\b': '서비스',
            r'\bprotection\b': '보호',
            r'\bprocessing\b': '처리',
            r'\bcollection\b': '수집',
            r'\bprovision\b': '제공',
            r'\bconsent\b': '동의',
            r'\bpersonal\b': '개인',
            r'\belectronic\b': '전자',
            r'\bauthentication\b': '인증',
            r'\bencryption\b': '암호화',
            r'\bnetwork\b': '네트워크'
        }
    
    def _clean_korean_text(self, text: str) -> str:
        """한국어 텍스트 정리"""
        
        # 한자/영어 -> 한국어 변환
        for pattern, replacement in self.korean_cleanup_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 나머지 한자 문자 제거
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        
        # 괄호 밖의 단독 영어 단어 제거
        text = re.sub(r'\b[A-Za-z]+\b(?!\))', '', text)
        
        # 특수문자 정리
        text = re.sub(r'[:：]\s*', ': ', text)  # 콜론 정리
        
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def _build_extraction_patterns(self) -> Dict[str, List[str]]:
        """답변 추출 패턴"""
        patterns = {
            "explicit_answer": [
                r'정답[:\s]*([1-5])',
                r'답[:\s]*([1-5])',
                r'최종\s*답[:\s]*([1-5])',
                r'결론[:\s]*([1-5])',
                r'따라서[^.]*?([1-5])번',
                r'그러므로[^.]*?([1-5])번',
                r'분석\s*결과[^.]*?([1-5])번',
                r'선택[:\s]*([1-5])',
            ],
            "choice_reference": [
                r'([1-5])번이\s*(?:정답|맞|적절|옳|해당)',
                r'선택지\s*([1-5])',
                r'([1-5])번을\s*선택',
                r'([1-5])번\s*항목',
                r'([1-5]):\s*[^1-5]*?(?:정답|맞|적절|옳)',
                r'가장\s*적절한\s*것은\s*([1-5])',
                r'올바른\s*답은\s*([1-5])',
            ],
            "reasoning_conclusion": [
                r'결론적으로[^.]*?([1-5])',
                r'종합하면[^.]*?([1-5])',
                r'판단하건대[^.]*?([1-5])',
                r'분석\s*결과[^.]*?([1-5])',
                r'검토\s*결과[^.]*?([1-5])',
                r'평가하면[^.]*?([1-5])',
            ],
            "high_confidence": [
                r'명확히\s*([1-5])번',
                r'확실히\s*([1-5])번',
                r'분명히\s*([1-5])번',
                r'당연히\s*([1-5])번',
            ],
            "contextual_answer": [
                r'(?:이\s*문제의\s*)?정답은\s*([1-5])',
                r'답변은\s*([1-5])번',
                r'해답은\s*([1-5])',
                r'정확한\s*답은\s*([1-5])',
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
        """검증 규칙 (한국어 품질 포함)"""
        rules = {
            "choice_range": lambda x: x.isdigit() and 1 <= int(x) <= 5,
            "length_appropriate": lambda x: 1 <= len(x) <= 3000,
            "not_empty": lambda x: x.strip() != "",
            "meaningful_content": lambda x: len(x.split()) >= 3 if not x.isdigit() else True,
            "korean_content": lambda x: bool(re.search(r'[가-힣]', x)),
            "no_chinese_chars": lambda x: not bool(re.search(r'[\u4e00-\u9fff]', x)),
            "minimal_english": lambda x: len(re.findall(r'[A-Za-z]', x)) < len(x) * 0.3,
            "professional_content": lambda x: any(term in x for term in ['법', '규정', '보안', '관리', '정책']) if len(x) > 50 else True,
            "no_repetition": lambda x: len(set(x.split())) / len(x.split()) > 0.7 if len(x.split()) > 10 else True,
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
            r"아닌\s*것",
            r"거짓인\s*것",
            r"맞지\s*않는",
            r"관련이\s*없는",
            r"해당되지\s*않는"
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
        """빠른 객관식 답변 추출 (한국어 정리 포함)"""
        
        # 한국어 정리 먼저 수행
        cleaned_response = self._clean_korean_text(response)
        
        # 캐시 확인
        response_hash = hash(cleaned_response[:100])
        if response_hash in self.pattern_cache:
            return self.pattern_cache[response_hash]
        
        # 우선순위별 패턴 확인
        for category in ["explicit_answer", "high_confidence", "choice_reference", 
                        "contextual_answer", "reasoning_conclusion"]:
            patterns = self.compiled_patterns.get(category, [])
            for pattern in patterns:
                match = pattern.search(cleaned_response)
                if match:
                    answer = match.group(1)
                    if self.validation_rules["choice_range"](answer):
                        # 캐시 저장
                        self.pattern_cache[response_hash] = answer
                        return answer
        
        # 위치 기반 숫자 검색
        numbers = re.findall(r'[1-5]', cleaned_response)
        if numbers:
            # 마지막 숫자 우선
            answer = numbers[-1]
            self.pattern_cache[response_hash] = answer
            return answer
        
        # 통계적 기본값
        return "3"
    
    def extract_answer_intelligently(self, response: str, question: str) -> ProcessedAnswer:
        """지능형 답변 추출 (한국어 정리 포함)"""
        
        # 한국어 정리
        cleaned_response = self._clean_korean_text(response)
        question_structure = self.analyze_question_structure(question)
        
        if question_structure["question_type"] == "multiple_choice":
            return self._extract_mc_answer_optimized(cleaned_response, question_structure)
        else:
            return self._extract_subjective_answer_optimized(cleaned_response, question_structure)
    
    def _extract_mc_answer_optimized(self, response: str, question_structure: Dict) -> ProcessedAnswer:
        """최적화 객관식 답변 추출"""
        extraction_results = []
        
        # 가중치 기반 패턴 매칭
        pattern_weights = {
            "explicit_answer": 1.0,
            "high_confidence": 0.9,
            "choice_reference": 0.8,
            "contextual_answer": 0.7,
            "reasoning_conclusion": 0.6
        }
        
        for method, patterns in self.compiled_patterns.items():
            weight = pattern_weights.get(method, 0.5)
            
            for i, pattern in enumerate(patterns):
                matches = pattern.finditer(response)
                for match in matches:
                    answer = match.group(1)
                    if self.validation_rules["choice_range"](answer):
                        # 위치 기반 보너스
                        position_bonus = match.start() / len(response) * 0.2
                        
                        # 패턴 순서 보너스
                        pattern_bonus = (len(patterns) - i) / len(patterns) * 0.1
                        
                        confidence = weight + position_bonus + pattern_bonus
                        
                        extraction_results.append({
                            "answer": answer,
                            "confidence": min(confidence, 1.0),
                            "method": method,
                            "position": match.start()
                        })
        
        if extraction_results:
            # 최고 신뢰도 결과 선택
            best_result = max(extraction_results, key=lambda x: x["confidence"])
            
            # 부정형 문제 보정
            if question_structure.get("has_negative", False):
                best_result["confidence"] *= 0.9
            
            return ProcessedAnswer(
                final_answer=best_result["answer"],
                confidence=best_result["confidence"],
                extraction_method=best_result["method"],
                validation_passed=True
            )
        
        # 실패 시 통계적 폴백
        statistical_answer = self._get_statistical_fallback(question_structure)
        return ProcessedAnswer(
            final_answer=statistical_answer,
            confidence=0.25,
            extraction_method="statistical_fallback",
            validation_passed=False
        )
    
    def _get_statistical_fallback(self, question_structure: Dict) -> str:
        """통계적 폴백 답변"""
        # 도메인별 선호 답변
        domain_preferences = {
            "개인정보보호": {"2": 0.4, "1": 0.3, "3": 0.2},
            "전자금융": {"2": 0.35, "3": 0.3, "1": 0.25},
            "정보보안": {"3": 0.4, "2": 0.35, "4": 0.15},
            "법령": {"3": 0.35, "2": 0.3, "1": 0.25}
        }
        
        # 도메인 기반 선택
        for domain in question_structure.get("domain_hints", []):
            if domain in domain_preferences:
                preferences = domain_preferences[domain]
                return max(preferences.items(), key=lambda x: x[1])[0]
        
        # 부정형 문제 처리
        if question_structure.get("has_negative", False):
            return "1"
        
        # 기본값
        return "3"
    
    def _extract_subjective_answer_optimized(self, response: str, 
                                          question_structure: Dict) -> ProcessedAnswer:
        """최적화 주관식 답변 추출 (한국어 품질 강화)"""
        
        # 접두사 제거
        cleaned_response = re.sub(
            r"^(답변|응답|해답|설명|분석|결론)[:\s]*", "", 
            response, 
            flags=re.IGNORECASE
        )
        
        # 구조화
        cleaned_response = self._structure_subjective_answer(
            cleaned_response, question_structure
        )
        
        # 한국어 품질 재확인 및 정리
        cleaned_response = self._final_korean_cleanup(cleaned_response)
        
        # 품질 평가
        confidence = self._evaluate_subjective_quality(
            cleaned_response, question_structure
        )
        
        return ProcessedAnswer(
            final_answer=cleaned_response.strip(),
            confidence=confidence,
            extraction_method="subjective_processing",
            validation_passed=confidence > 0.4
        )
    
    def _final_korean_cleanup(self, text: str) -> str:
        """최종 한국어 정리"""
        
        # 한자 및 외국어 확인 후 제거/변환
        text = self._clean_korean_text(text)
        
        # 한국어 비율 확인
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars > 0 and korean_chars / total_chars < 0.6:
            # 한국어 비율이 낮으면 한국어 문장만 추출
            sentences = re.split(r'[.!?]\s+', text)
            korean_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    sent_korean = len(re.findall(r'[가-힣]', sentence))
                    sent_total = len(re.sub(r'[^\w]', '', sentence))
                    
                    if sent_total > 0 and sent_korean / sent_total > 0.5:
                        korean_sentences.append(sentence)
            
            if korean_sentences:
                text = '. '.join(korean_sentences)
                if not text.endswith('.'):
                    text += '.'
        
        return text
    
    def _structure_subjective_answer(self, response: str, 
                                           question_structure: Dict) -> str:
        """주관식 답변 구조화"""
        
        # 문장 분리 및 중복 제거
        sentences = re.split(r'[.!?]\s+', response)
        unique_sentences = []
        seen_keys = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 15:
                # 의미적 중복 체크
                key = sentence[:30].lower()
                if key not in seen_keys:
                    unique_sentences.append(sentence)
                    seen_keys.add(key)
        
        # 문장 재조립
        structured_response = '. '.join(unique_sentences)
        if structured_response and not structured_response.endswith('.'):
            structured_response += '.'
        
        # 도메인별 맞춤 보강 (한국어로)
        if "개인정보보호" in question_structure.get("domain_hints", []):
            if len(structured_response) < 80:
                structured_response += " 개인정보보호법에 따른 추가적인 보호 조치가 필요합니다."
        elif "전자금융" in question_structure.get("domain_hints", []):
            if len(structured_response) < 80:
                structured_response += " 전자금융거래법에 따른 안전성 확보 방안이 요구됩니다."
        
        # 길이 조정
        if len(structured_response) < 60:
            structured_response = f"해당 사항과 관련하여 {structured_response}"
        elif len(structured_response) > 1200:
            # 중요한 문장 우선 유지
            sentences = structured_response.split('. ')
            important_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in ['법', '규정', '필수', '반드시', '중요']):
                    important_sentences.append(sentence)
                elif len('. '.join(important_sentences)) < 800:
                    important_sentences.append(sentence)
                else:
                    break
            
            structured_response = '. '.join(important_sentences)
            if not structured_response.endswith('.'):
                structured_response += '.'
        
        return structured_response
    
    def _evaluate_subjective_quality(self, response: str, 
                                            question_structure: Dict) -> float:
        """주관식 품질 평가 (한국어 품질 강화)"""
        confidence = 0.4  # 기본값
        
        # 길이 평가
        length = len(response)
        if 150 <= length <= 800:
            confidence += 0.25
        elif 80 <= length < 150 or 800 < length <= 1200:
            confidence += 0.15
        elif length < 50:
            confidence -= 0.15
        
        # 한국어 품질 평가
        korean_chars = len(re.findall(r'[가-힣]', response))
        total_chars = len(re.sub(r'[^\w]', '', response))
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            if korean_ratio > 0.8:
                confidence += 0.2
            elif korean_ratio > 0.6:
                confidence += 0.1
            elif korean_ratio < 0.4:
                confidence -= 0.2
        
        # 한자/외국어 페널티
        if re.search(r'[\u4e00-\u9fff]', response):
            confidence -= 0.3
        
        if len(re.findall(r'[A-Za-z]', response)) > len(response) * 0.2:
            confidence -= 0.15
        
        # 전문성 평가
        professional_terms = {
            "법령": ["법", "조", "항", "규정", "시행령"],
            "보안": ["보안", "암호화", "인증", "접근제어", "관리"],
            "정책": ["정책", "절차", "방안", "대책", "조치"],
            "품질": ["적절한", "체계적", "효과적", "지속적", "종합적"]
        }
        
        for category, terms in professional_terms.items():
            term_count = sum(1 for term in terms if term in response)
            confidence += min(term_count * 0.05, 0.15)
        
        # 구조적 품질
        structure_bonus = 0
        if re.search(r'첫째|둘째|셋째|1\)|2\)|3\)', response):
            structure_bonus += 0.1
        if re.search(r'따라서|그러므로|결론적으로', response):
            structure_bonus += 0.05
        if re.search(r'예를\s*들어|구체적으로', response):
            structure_bonus += 0.05
        
        confidence += min(structure_bonus, 0.2)
        
        # 도메인 관련성
        domain_hints = question_structure.get("domain_hints", [])
        if domain_hints:
            domain_terms = {
                "개인정보보호": ["개인정보", "정보주체", "동의", "수집"],
                "전자금융": ["전자금융", "전자적", "거래", "보안"],
                "정보보안": ["정보보안", "보안관리", "접근통제"]
            }
            
            for domain in domain_hints:
                if domain in domain_terms:
                    matched_terms = sum(1 for term in domain_terms[domain] if term in response)
                    confidence += min(matched_terms * 0.03, 0.12)
        
        # 복잡도 대비 적절성
        complexity = question_structure.get("complexity_score", 0.5)
        if complexity > 0.7 and length > 200:
            confidence += 0.1
        elif complexity < 0.3 and 80 <= length <= 300:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def post_process_answer(self, raw_response: str, question: str,
                          question_type: str) -> str:
        """통합 후처리 함수 (한국어 강화)"""
        
        # 먼저 한국어 정리
        cleaned_response = self._clean_korean_text(raw_response)
        
        if question_type == "multiple_choice":
            # 빠른 추출 시도
            quick_answer = self.extract_mc_answer_fast(cleaned_response)
            if quick_answer and quick_answer != "3":
                return quick_answer
            
            # 상세 추출
            processed = self.extract_answer_intelligently(cleaned_response, question)
            return processed.final_answer
        else:
            # 주관식 처리
            processed = self.extract_answer_intelligently(cleaned_response, question)
            
            if self.validate_final_answer(processed, question, question_type):
                return processed.final_answer
            else:
                # 폴백
                return self._generate_domain_specific_fallback(question)
    
    def _generate_domain_specific_fallback(self, question: str) -> str:
        """도메인별 맞춤 폴백 생성 (한국어만)"""
        question_structure = self.analyze_question_structure(question)
        domain_hints = question_structure.get("domain_hints", [])
        
        if "개인정보보호" in domain_hints:
            return "개인정보보호법에 따른 체계적인 개인정보 관리 방안 수립과 정보주체 권리 보장을 위한 적절한 절차 마련이 필요합니다."
        elif "전자금융" in domain_hints:
            return "전자금융거래법에 따른 전자적 장치의 보안성 확보와 안전한 전자금융거래 환경 조성을 위한 종합적 대책이 요구됩니다."
        elif "정보보안" in domain_hints:
            return "정보보안 관리체계 구축을 통한 체계적 보안 관리와 지속적인 위험 평가 및 개선 방안 수립이 필요합니다."
        else:
            return "관련 법령과 규정에 따른 적절한 관리 방안 수립과 지속적인 개선을 통한 체계적 대응이 필요합니다."
    
    def validate_final_answer(self, processed_answer: ProcessedAnswer,
                            question: str, question_type: str) -> bool:
        """최종 답변 검증 (한국어 품질 포함)"""
        
        answer = processed_answer.final_answer
        
        # 기본 검증
        if not self.validation_rules["not_empty"](answer):
            return False
        
        # 한국어 품질 검증
        if not self.validation_rules["korean_content"](answer):
            return False
        
        if not self.validation_rules["no_chinese_chars"](answer):
            return False
        
        if not self.validation_rules["minimal_english"](answer):
            return False
        
        if question_type == "multiple_choice":
            return self.validation_rules["choice_range"](answer)
        else:
            # 주관식 검증
            validations = [
                self.validation_rules["length_appropriate"](answer),
                self.validation_rules["meaningful_content"](answer),
                self.validation_rules["korean_content"](answer),
                self.validation_rules["professional_content"](answer),
                self.validation_rules["no_repetition"](answer),
                self.validation_rules["no_chinese_chars"](answer),
                self.validation_rules["minimal_english"](answer)
            ]
            
            # 80% 이상 통과하면 유효
            return sum(validations) / len(validations) >= 0.8
    
    def get_processing_statistics(self) -> Dict:
        """처리 통계 반환"""
        return {
            "structure_cache_size": len(self.structure_cache),
            "pattern_cache_size": len(self.pattern_cache),
            "answer_statistics": self.answer_statistics,
            "cache_efficiency": {
                "structure_cache_hits": getattr(self, '_structure_cache_hits', 0),
                "pattern_cache_hits": getattr(self, '_pattern_cache_hits', 0)
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        self.structure_cache.clear()
        self.pattern_cache.clear()
        print("데이터 처리기 정리 완료")