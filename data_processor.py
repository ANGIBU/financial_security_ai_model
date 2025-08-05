# data_processor.py
"""
지능형 데이터 전처리 및 후처리 시스템 - 최적화 버전
정확도 향상을 위한 고급 패턴 매칭 및 검증
"""

import re
import pandas as pd
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

class IntelligentDataProcessor:
    """지능형 데이터 처리 클래스 - 최적화 버전"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.answer_extraction_patterns = self._build_extraction_patterns()
        self.validation_rules = self._build_validation_rules()
        
        # 성능 향상을 위한 컴파일된 정규식
        self.compiled_patterns = self._compile_patterns()
        
    def _build_extraction_patterns(self) -> Dict[str, List[str]]:
        """고급 답변 추출 패턴"""
        patterns = {
            "explicit_answer": [
                r'정답[:\s]*([1-5])',
                r'답[:\s]*([1-5])',
                r'최종\s*답[:\s]*([1-5])',
                r'결론[:\s]*([1-5])',
                r'따라서[^.]*?([1-5])번',
                r'그러므로[^.]*?([1-5])번',
                r'분석\s*결과[^.]*?([1-5])번',
            ],
            "choice_reference": [
                r'([1-5])번이\s*(?:정답|맞|적절|옳|해당)',
                r'선택지\s*([1-5])',
                r'([1-5])번을\s*선택',
                r'([1-5])번\s*항목',
                r'([1-5]):\s*[^1-5]*?(?:정답|맞|적절|옳)',
                r'가장\s*적절한\s*것은\s*([1-5])',
            ],
            "reasoning_conclusion": [
                r'결론적으로[^.]*?([1-5])',
                r'종합하면[^.]*?([1-5])',
                r'판단하건대[^.]*?([1-5])',
                r'분석\s*결과[^.]*?([1-5])',
                r'검토\s*결과[^.]*?([1-5])',
            ],
            "high_confidence": [
                r'명확히\s*([1-5])번',
                r'확실히\s*([1-5])번',
                r'분명히\s*([1-5])번',
            ]
        }
        return patterns
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """패턴 컴파일 (성능 향상)"""
        compiled = {}
        for category, patterns in self.answer_extraction_patterns.items():
            compiled[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled
    
    def _build_validation_rules(self) -> Dict[str, callable]:
        """답변 검증 규칙"""
        rules = {
            "choice_range": lambda x: x.isdigit() and 1 <= int(x) <= 5,
            "length_appropriate": lambda x: 1 <= len(x) <= 2000,
            "not_empty": lambda x: x.strip() != "",
            "meaningful_content": lambda x: len(x.split()) >= 3 if not x.isdigit() else True,
        }
        return rules
    
    def analyze_question_structure(self, question: str) -> Dict:
        """질문 구조 분석 - 최적화 버전"""
        lines = question.strip().split("\n")
        structure = {
            "question_text": "",
            "choices": [],
            "choice_count": 0,
            "has_negative": False,
            "question_type": "subjective"
        }
        
        question_parts = []
        choices = []
        
        # 컴파일된 선택지 패턴
        choice_patterns = [
            re.compile(r"^\s*([1-5])\s+(.+)"),
            re.compile(r"^\s*([1-5])[.)]\s*(.+)"),
            re.compile(r"^\s*([①-⑤])\s*(.+)"),
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
                        "text": choice_text.strip()
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
        
        return structure
    
    def _detect_negative_question(self, question_text: str) -> bool:
        """부정형 질문 감지 - 최적화"""
        negative_pattern = re.compile(
            r"해당하지\s*않는|적절하지\s*않은|옳지\s*않은|틀린\s*것|잘못된\s*것|부적절한|제외한\s*것|아닌\s*것",
            re.IGNORECASE
        )
        return bool(negative_pattern.search(question_text))
    
    def extract_mc_answer_fast(self, response: str) -> str:
        """객관식 답변 빠른 추출"""
        # 우선순위별 패턴 확인
        for category in ["explicit_answer", "high_confidence", "choice_reference", "reasoning_conclusion"]:
            patterns = self.compiled_patterns.get(category, [])
            for pattern in patterns:
                match = pattern.search(response)
                if match:
                    answer = match.group(1)
                    if self.validation_rules["choice_range"](answer):
                        return answer
        
        # 단순 숫자 검색
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            # 마지막에 나온 숫자를 선택 (보통 결론 부분)
            return numbers[-1]
        
        return "2"  # 기본값
    
    def extract_answer_intelligently(self, response: str, question: str) -> ProcessedAnswer:
        """지능형 답변 추출 - 최적화 버전"""
        question_structure = self.analyze_question_structure(question)
        
        if question_structure["question_type"] == "multiple_choice":
            return self._extract_mc_answer_optimized(response, question_structure)
        else:
            return self._extract_subjective_answer(response, question_structure)
    
    def _extract_mc_answer_optimized(self, response: str, question_structure: Dict) -> ProcessedAnswer:
        """최적화된 객관식 답변 추출"""
        extraction_results = []
        
        # 컴파일된 패턴으로 빠른 매칭
        for method, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(response)
                for match in matches:
                    answer = match.group(1)
                    if self.validation_rules["choice_range"](answer):
                        confidence = self._calculate_confidence_fast(method, match.start())
                        extraction_results.append({
                            "answer": answer,
                            "confidence": confidence,
                            "method": method,
                            "position": match.start()
                        })
        
        if extraction_results:
            # 위치와 신뢰도를 고려한 최적 선택
            best_result = max(extraction_results, 
                            key=lambda x: x["confidence"] - (x["position"] / len(response)) * 0.1)
            
            return ProcessedAnswer(
                final_answer=best_result["answer"],
                confidence=best_result["confidence"],
                extraction_method=best_result["method"],
                validation_passed=True
            )
        
        # 실패 시 빠른 폴백
        fallback = self.extract_mc_answer_fast(response)
        return ProcessedAnswer(
            final_answer=fallback,
            confidence=0.3,
            extraction_method="fallback",
            validation_passed=False
        )
    
    def _calculate_confidence_fast(self, method: str, position: int) -> float:
        """빠른 신뢰도 계산"""
        base_confidence = {
            "explicit_answer": 0.9,
            "high_confidence": 0.85,
            "choice_reference": 0.7,
            "reasoning_conclusion": 0.6
        }.get(method, 0.5)
        
        # 위치 보너스 (뒤쪽일수록 높음)
        position_bonus = min(position / 1000, 0.1)
        
        return min(base_confidence + position_bonus, 1.0)
    
    def _extract_subjective_answer(self, response: str, 
                                 question_structure: Dict) -> ProcessedAnswer:
        """주관식 답변 추출 - 최적화"""
        
        # 불필요한 접두사 제거
        cleaned_response = re.sub(
            r"^(답변|응답|해답|설명)[:\s]*", "", 
            response, 
            flags=re.IGNORECASE
        )
        
        # 구조화
        cleaned_response = self._structure_subjective_answer_fast(cleaned_response)
        
        # 빠른 품질 평가
        confidence = self._evaluate_subjective_quality_fast(cleaned_response)
        
        return ProcessedAnswer(
            final_answer=cleaned_response.strip(),
            confidence=confidence,
            extraction_method="subjective_processing",
            validation_passed=confidence > 0.3
        )
    
    def _structure_subjective_answer_fast(self, response: str) -> str:
        """주관식 답변 빠른 구조화"""
        
        # 중복 제거
        sentences = re.split(r'[.!?]\s+', response)
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                key = sentence[:30].lower()
                if key not in seen:
                    unique_sentences.append(sentence)
                    seen.add(key)
        
        response = '. '.join(unique_sentences)
        if response and not response.endswith('.'):
            response += '.'
        
        # 길이 조정
        if len(response) < 50:
            response = f"해당 사항은 {response}"
        elif len(response) > 800:
            response = response[:750] + "..."
        
        return response
    
    def _evaluate_subjective_quality_fast(self, response: str) -> float:
        """주관식 품질 빠른 평가"""
        confidence = 0.4  # 기본값
        
        # 길이 체크
        length = len(response)
        if 100 <= length <= 600:
            confidence += 0.2
        
        # 전문 용어 체크 (간소화)
        professional_count = sum(1 for term in ['개인정보', '보안', '시스템', '정책'] 
                               if term in response)
        confidence += min(professional_count * 0.1, 0.3)
        
        return min(confidence, 1.0)
    
    def post_process_answer(self, raw_response: str, question: str,
                          question_type: str) -> str:
        """통합 후처리 함수 - 최적화"""
        
        if question_type == "multiple_choice":
            # 빠른 추출 우선 시도
            quick_answer = self.extract_mc_answer_fast(raw_response)
            if quick_answer and quick_answer != "2":  # 기본값이 아니면
                return quick_answer
            
            # 상세 추출
            processed = self.extract_answer_intelligently(raw_response, question)
            return processed.final_answer
        else:
            # 주관식 처리
            processed = self.extract_answer_intelligently(raw_response, question)
            
            if self.validate_final_answer(processed, question, question_type):
                return processed.final_answer
            else:
                return "해당 문제에 대한 전문적인 분석이 필요한 사안입니다."
    
    def validate_final_answer(self, processed_answer: ProcessedAnswer,
                            question: str, question_type: str) -> bool:
        """최종 답변 검증 - 간소화"""
        
        answer = processed_answer.final_answer
        
        if not self.validation_rules["not_empty"](answer):
            return False
        
        if question_type == "multiple_choice":
            return self.validation_rules["choice_range"](answer)
        else:
            return self.validation_rules["length_appropriate"](answer)