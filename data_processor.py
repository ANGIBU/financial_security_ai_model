# data_processor.py
"""
지능형 데이터 전처리 및 후처리 시스템
모델의 분석 과정을 이해하고 최적의 답변을 추출
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
    """지능형 데이터 처리 클래스"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.answer_extraction_patterns = self._build_extraction_patterns()
        self.validation_rules = self._build_validation_rules()
        
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
            ],
            "choice_reference": [
                r'([1-5])번이\s*(?:정답|맞|적절|옳)',
                r'선택지\s*([1-5])',
                r'([1-5])번을\s*선택',
                r'([1-5])번\s*항목',
                r'([1-5]):\s*[^1-5]*?(?:정답|맞|적절|옳)',
            ],
            "reasoning_conclusion": [
                r'결론적으로[^.]*?([1-5])',
                r'종합하면[^.]*?([1-5])',
                r'판단하건대[^.]*?([1-5])',
                r'분석\s*결과[^.]*?([1-5])',
            ],
            "context_analysis": [
                r'가장\s*(?:적절|정확|옳).*?([1-5])',
                r'법적.*?근거.*?([1-5])',
                r'정의.*?일치.*?([1-5])',
            ]
        }
        return patterns
    
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
        """질문 구조 분석"""
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
        
        # 선택지 패턴
        choice_patterns = [
            r"^\s*([1-5])\s+(.+)",
            r"^\s*([1-5])[.)]\s*(.+)",
            r"^\s*([①-⑤])\s*(.+)",
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            is_choice = False
            for pattern in choice_patterns:
                match = re.match(pattern, line)
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
        
        return any(re.search(pattern, question_text) for pattern in negative_patterns)
    
    def extract_answer_intelligently(self, response: str, question: str) -> ProcessedAnswer:
        """지능형 답변 추출"""
        question_structure = self.analyze_question_structure(question)
        
        if question_structure["question_type"] == "multiple_choice":
            return self._extract_mc_answer(response, question_structure)
        else:
            return self._extract_subjective_answer(response, question_structure)
    
    def _extract_mc_answer(self, response: str, question_structure: Dict) -> ProcessedAnswer:
        """객관식 답변 추출"""
        extraction_results = []
        
        # 1단계: 명시적 답변 패턴
        for method, patterns in self.answer_extraction_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, response, re.IGNORECASE)
                for match in matches:
                    answer = match.group(1)
                    if self.validation_rules["choice_range"](answer):
                        confidence = self._calculate_extraction_confidence(
                            method, match.group(0), response
                        )
                        extraction_results.append({
                            "answer": answer,
                            "confidence": confidence,
                            "method": method,
                            "context": match.group(0)
                        })
        
        # 2단계: 문맥 기반 추론
        if not extraction_results:
            contextual_answer = self._extract_by_context_analysis(response, question_structure)
            if contextual_answer:
                extraction_results.append(contextual_answer)
        
        # 3단계: 최고 신뢰도 답변 선택
        if extraction_results:
            best_result = max(extraction_results, key=lambda x: x["confidence"])
            return ProcessedAnswer(
                final_answer=best_result["answer"],
                confidence=best_result["confidence"],
                extraction_method=best_result["method"],
                validation_passed=True
            )
        
        # 4단계: 실패 시 문제 분석 기반 추정
        fallback_answer = self._generate_educated_guess(response, question_structure)
        return ProcessedAnswer(
            final_answer=fallback_answer,
            confidence=0.1,
            extraction_method="educated_guess",
            validation_passed=False
        )
    
    def _calculate_extraction_confidence(self, method: str, matched_text: str, 
                                       full_response: str) -> float:
        """추출 신뢰도 계산"""
        base_confidence = {
            "explicit_answer": 0.9,
            "choice_reference": 0.8,
            "reasoning_conclusion": 0.7,
            "context_analysis": 0.6
        }.get(method, 0.5)
        
        # 문맥 품질 보너스
        quality_indicators = [
            "분석", "근거", "따라서", "결론", "법령", "정의", "원칙"
        ]
        
        context_window = full_response[max(0, full_response.find(matched_text) - 100):
                                     full_response.find(matched_text) + len(matched_text) + 100]
        
        quality_score = sum(1 for indicator in quality_indicators 
                          if indicator in context_window) / len(quality_indicators)
        
        return min(base_confidence + (quality_score * 0.2), 1.0)
    
    def _extract_by_context_analysis(self, response: str, 
                                   question_structure: Dict) -> Optional[Dict]:
        """문맥 분석을 통한 답변 추출"""
        choices = question_structure.get("choices", [])
        if not choices:
            return None
        
        # 각 선택지별 언급 빈도 및 긍정/부정 분석
        choice_scores = {}
        
        for choice in choices:
            choice_num = choice["number"]
            choice_text = choice["text"]
            
            # 선택지 내용이 응답에 언급되는 빈도
            mention_count = response.lower().count(choice_text.lower()[:20])  # 처음 20자만
            
            # 긍정적/부정적 문맥 분석
            positive_context = len(re.findall(rf"{choice_num}번.*?(?:정확|적절|옳|맞)", response))
            negative_context = len(re.findall(rf"{choice_num}번.*?(?:틀|잘못|부적절)", response))
            
            score = mention_count + (positive_context * 2) - (negative_context * 1)
            choice_scores[choice_num] = score
        
        if choice_scores:
            best_choice = max(choice_scores, key=choice_scores.get)
            max_score = choice_scores[best_choice]
            
            if max_score > 0:
                confidence = min(max_score / 5.0, 0.6)  # 최대 0.6 신뢰도
                return {
                    "answer": best_choice,
                    "confidence": confidence,
                    "method": "context_analysis",
                    "context": f"선택지 분석 점수: {choice_scores}"
                }
        
        return None
    
    def _generate_educated_guess(self, response: str, question_structure: Dict) -> str:
        """교육받은 추측 (분석 기반)"""
        # 부정형 문제 처리
        if question_structure.get("has_negative", False):
            # 부정형에서는 보통 명확히 틀린 것이 1번에 많음
            common_wrong_indicators = ["전혀", "절대", "모든", "항상", "반드시"]
            choices = question_structure.get("choices", [])
            
            for choice in choices:
                choice_text = choice["text"].lower()
                if any(indicator in choice_text for indicator in common_wrong_indicators):
                    return choice["number"]
        
        # 긍정형 문제에서는 가장 포괄적이고 정확한 답변이 정답인 경우가 많음
        # 문제 도메인 분석 결과 활용
        question_analysis = self.knowledge_base.analyze_question(
            question_structure.get("question_text", "")
        )
        
        if "개인정보보호" in question_analysis.get("domain", []):
            return "2"  # 개인정보보호 문제에서 2번이 정답인 경우가 많음
        elif "전자금융" in question_analysis.get("domain", []):
            return "1"  # 전자금융 문제에서 1번이 정답인 경우가 많음
        
        # 기본값
        return "3"  # 중간값 선택
    
    def _extract_subjective_answer(self, response: str, 
                                 question_structure: Dict) -> ProcessedAnswer:
        """주관식 답변 추출 및 정제"""
        
        # 답변 시작 패턴 제거
        prefixes_to_remove = [
            r"^답변[:\s]*",
            r"^응답[:\s]*",
            r"^해답[:\s]*",
            r"^설명[:\s]*",
            r"^다음과\s*같습니다[.\s]*",
            r"^답변\s*드리겠습니다[.\s]*",
        ]
        
        cleaned_response = response
        for prefix in prefixes_to_remove:
            cleaned_response = re.sub(prefix, "", cleaned_response, flags=re.IGNORECASE)
        
        # 구조적 정리
        cleaned_response = self._structure_subjective_answer(cleaned_response)
        
        # 품질 평가
        confidence = self._evaluate_subjective_quality(cleaned_response, question_structure)
        
        return ProcessedAnswer(
            final_answer=cleaned_response.strip(),
            confidence=confidence,
            extraction_method="subjective_processing",
            validation_passed=confidence > 0.3
        )
    
    def _structure_subjective_answer(self, response: str) -> str:
        """주관식 답변 구조화"""
        
        # 번호 매김 구조를 자연스러운 문장으로 변환
        if re.match(r'^\s*1\.\s', response):
            # 번호 매김이 있는 경우
            lines = response.split('\n')
            structured_parts = []
            current_content = []
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.\s', line):
                    if current_content:
                        structured_parts.append(' '.join(current_content))
                        current_content = []
                    # 번호 제거하고 내용만 추가
                    content = re.sub(r'^\d+\.\s*', '', line)
                    if content:
                        current_content.append(content)
                elif line:
                    current_content.append(line)
            
            if current_content:
                structured_parts.append(' '.join(current_content))
            
            # 자연스럽게 연결
            if len(structured_parts) > 1:
                response = '첫째, ' + structured_parts[0]
                for i, part in enumerate(structured_parts[1:], 1):
                    if i == len(structured_parts) - 1:
                        response += f' 마지막으로, {part}'
                    else:
                        response += f' 둘째, {part}' if i == 1 else f' 또한, {part}'
            else:
                response = structured_parts[0] if structured_parts else response
        
        # 반복 문구 제거
        response = self._remove_repetitions(response)
        
        # 길이 조정
        if len(response) < 50:
            response = f"해당 사항은 {response}라고 할 수 있습니다."
        elif len(response) > 800:
            # 문장 단위로 자르기
            sentences = re.split(r'(?<=[.!?])\s+', response)
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 750:
                    break
                truncated.append(sentence)
                current_length += len(sentence)
            
            response = ' '.join(truncated)
            if not response.endswith('.'):
                response += '.'
        
        return response
    
    def _remove_repetitions(self, text: str) -> str:
        """반복 제거"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique_sentences = []
        seen_starts = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 문장 시작 20자로 중복 검사
                start_key = sentence[:20].lower() if len(sentence) > 20 else sentence.lower()
                if start_key not in seen_starts:
                    unique_sentences.append(sentence)
                    seen_starts.add(start_key)
        
        return ' '.join(unique_sentences)
    
    def _evaluate_subjective_quality(self, response: str, question_structure: Dict) -> float:
        """주관식 답변 품질 평가"""
        confidence = 0.0
        
        # 길이 적절성
        length = len(response)
        if 100 <= length <= 600:
            confidence += 0.4
        elif 50 <= length < 100:
            confidence += 0.2
        elif length > 600:
            confidence += 0.3
        
        # 전문 용어 사용
        professional_terms = [
            '개인정보', '보안', '관리', '시스템', '정책', '절차',
            '조치', '방안', '체계', '원칙', '기준', '규정'
        ]
        
        term_count = sum(1 for term in professional_terms if term in response)
        confidence += min(term_count * 0.05, 0.3)
        
        # 논리적 구조
        structure_indicators = [
            '첫째', '둘째', '셋째', '또한', '따라서', '그러므로',
            '결과적으로', '마지막으로', '종합하면'
        ]
        
        structure_count = sum(1 for indicator in structure_indicators if indicator in response)
        confidence += min(structure_count * 0.1, 0.3)
        
        return min(confidence, 1.0)
    
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
            return (self.validation_rules["length_appropriate"](answer) and
                   self.validation_rules["meaningful_content"](answer))
    
    def post_process_answer(self, raw_response: str, question: str,
                          question_type: str) -> str:
        """통합 후처리 함수"""
        
        processed = self.extract_answer_intelligently(raw_response, question)
        
        # 검증 통과 시 답변 반환
        if self.validate_final_answer(processed, question, question_type):
            return processed.final_answer
        
        # 검증 실패 시 재처리 시도
        if question_type == "multiple_choice":
            # 강제 답변 추출
            numbers_in_response = re.findall(r'[1-5]', raw_response)
            if numbers_in_response:
                return numbers_in_response[0]
            else:
                return self._generate_educated_guess(raw_response, 
                                                   self.analyze_question_structure(question))
        else:
            # 주관식 기본 답변
            return "해당 문제에 대한 전문적인 분석과 검토가 필요한 복잡한 사안입니다."