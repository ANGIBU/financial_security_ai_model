# pattern_learner.py
"""
답변 패턴 학습 및 예측 시스템
과거 데이터에서 패턴을 학습하여 정확도 향상
"""

import re
import json
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np

class AnswerPatternLearner:
    """답변 패턴 학습 클래스"""
    
    def __init__(self):
        self.patterns = {
            "keyword_answer_map": defaultdict(Counter),
            "domain_answer_distribution": defaultdict(Counter),
            "negative_answer_patterns": Counter(),
            "length_answer_correlation": defaultdict(list),
            "law_mention_answers": defaultdict(Counter)
        }
        
        # 학습된 규칙
        self.learned_rules = self._initialize_rules()
        
    def _initialize_rules(self) -> Dict:
        """초기 규칙 설정 (도메인 지식 기반)"""
        return {
            "개인정보_정의": {
                "keywords": ["개인정보", "정의", "의미"],
                "preferred_answers": {"2": 0.6, "1": 0.2, "3": 0.1},
                "confidence": 0.7
            },
            "전자금융_정의": {
                "keywords": ["전자금융", "정의", "전자적"],
                "preferred_answers": {"2": 0.5, "1": 0.3, "3": 0.1},
                "confidence": 0.6
            },
            "유출_신고": {
                "keywords": ["유출", "신고", "즉시", "지체"],
                "preferred_answers": {"1": 0.7, "2": 0.2},
                "confidence": 0.8
            },
            "암호화_필수": {
                "keywords": ["암호화", "필수", "반드시"],
                "preferred_answers": {"1": 0.4, "2": 0.4, "3": 0.1},
                "confidence": 0.5
            },
            "부정형_일반": {
                "keywords": ["해당하지 않는", "적절하지 않은", "옳지 않은"],
                "preferred_answers": {"1": 0.5, "5": 0.2, "4": 0.15},
                "confidence": 0.6
            },
            "법령_조항": {
                "keywords": ["법", "제", "조", "항", "규정"],
                "preferred_answers": {"2": 0.35, "3": 0.3, "1": 0.2},
                "confidence": 0.5
            },
            "보안_조치": {
                "keywords": ["보안", "조치", "대책", "방안"],
                "preferred_answers": {"2": 0.4, "3": 0.3, "1": 0.2},
                "confidence": 0.55
            },
            "관리체계": {
                "keywords": ["관리체계", "ISMS", "정보보호"],
                "preferred_answers": {"3": 0.4, "2": 0.35, "1": 0.15},
                "confidence": 0.6
            }
        }
    
    def analyze_question_pattern(self, question: str) -> Dict:
        """문제 패턴 분석"""
        question_lower = question.lower()
        
        # 매칭된 규칙 찾기
        matched_rules = []
        for rule_name, rule_info in self.learned_rules.items():
            keywords = rule_info["keywords"]
            match_count = sum(1 for kw in keywords if kw.lower() in question_lower)
            
            if match_count >= 2:  # 2개 이상 키워드 매칭
                matched_rules.append({
                    "rule": rule_name,
                    "match_score": match_count / len(keywords),
                    "confidence": rule_info["confidence"],
                    "answers": rule_info["preferred_answers"]
                })
        
        # 가장 높은 매칭 점수의 규칙 선택
        if matched_rules:
            best_rule = max(matched_rules, key=lambda x: x["match_score"] * x["confidence"])
            return best_rule
        
        return None
    
    def predict_answer(self, question: str, structure: Dict) -> Tuple[str, float]:
        """패턴 기반 답변 예측"""
        
        # 부정형 문제 특별 처리
        if structure.get("has_negative", False):
            return self._predict_negative_answer(question)
        
        # 패턴 분석
        pattern_match = self.analyze_question_pattern(question)
        
        if pattern_match:
            # 가중치 기반 답변 선택
            answers = pattern_match["answers"]
            best_answer = max(answers.items(), key=lambda x: x[1])
            
            # 신뢰도 계산
            confidence = pattern_match["confidence"] * pattern_match["match_score"]
            
            return best_answer[0], confidence
        
        # 기본 예측 (통계 기반)
        return self._statistical_prediction(question, structure)
    
    def _predict_negative_answer(self, question: str) -> Tuple[str, float]:
        """부정형 문제 예측"""
        question_lower = question.lower()
        
        # 특정 패턴 체크
        if "모든" in question_lower or "항상" in question_lower:
            return "1", 0.7  # 극단적 표현은 보통 틀림
        
        if "제외" in question_lower:
            return "5", 0.6  # 마지막 선택지가 제외 대상인 경우 많음
        
        # 기본 부정형 예측
        rule = self.learned_rules["부정형_일반"]
        answers = rule["preferred_answers"]
        best_answer = max(answers.items(), key=lambda x: x[1])
        
        return best_answer[0], rule["confidence"]
    
    def _statistical_prediction(self, question: str, structure: Dict) -> Tuple[str, float]:
        """통계 기반 예측"""
        
        # 문제 길이 기반
        question_length = len(question)
        
        if structure["question_type"] == "multiple_choice":
            # 길이별 선호 답변
            if question_length < 200:
                return "2", 0.4  # 짧은 문제는 2번
            elif question_length < 400:
                return "3", 0.35  # 중간 길이는 3번
            else:
                return "2", 0.35  # 긴 문제는 다시 2번
        else:
            return "", 0.1  # 주관식은 예측 불가
    
    def update_patterns(self, question: str, correct_answer: str, structure: Dict):
        """패턴 업데이트 (학습)"""
        
        # 키워드-답변 매핑 업데이트
        keywords = re.findall(r'[가-힣]+', question.lower())
        for keyword in keywords:
            if len(keyword) >= 2:  # 2글자 이상만
                self.patterns["keyword_answer_map"][keyword][correct_answer] += 1
        
        # 도메인별 답변 분포
        if "개인정보" in question:
            self.patterns["domain_answer_distribution"]["개인정보"][correct_answer] += 1
        if "전자금융" in question:
            self.patterns["domain_answer_distribution"]["전자금융"][correct_answer] += 1
        
        # 부정형 패턴
        if structure.get("has_negative", False):
            self.patterns["negative_answer_patterns"][correct_answer] += 1
        
        # 길이-답변 상관관계
        length_category = len(question) // 100 * 100  # 100 단위로 분류
        self.patterns["length_answer_correlation"][length_category].append(correct_answer)
    
    def get_confidence_boost(self, question: str, predicted_answer: str) -> float:
        """신뢰도 부스트 계산"""
        
        # 패턴 매칭 확인
        pattern_match = self.analyze_question_pattern(question)
        
        if pattern_match:
            answers = pattern_match["answers"]
            if predicted_answer in answers:
                # 패턴과 일치하면 부스트
                return answers[predicted_answer] * 0.2
        
        return 0.0
    
    def save_patterns(self, filepath: str = "./learned_patterns.pkl"):
        """학습된 패턴 저장"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                "patterns": self.patterns,
                "rules": self.learned_rules
            }, f)
    
    def load_patterns(self, filepath: str = "./learned_patterns.pkl"):
        """학습된 패턴 로드"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.patterns = data["patterns"]
                self.learned_rules = data["rules"]
                return True
        except:
            return False

class SmartAnswerSelector:
    """스마트 답변 선택기"""
    
    def __init__(self):
        self.pattern_learner = AnswerPatternLearner()
        self.selection_history = []
        
    def select_best_answer(self, question: str, model_response: str, 
                         structure: Dict, confidence: float) -> Tuple[str, float]:
        """최적 답변 선택"""
        
        # 1. 모델 응답에서 답변 추출
        extracted_answers = self._extract_all_possible_answers(model_response)
        
        if not extracted_answers:
            # 패턴 기반 예측
            pattern_answer, pattern_conf = self.pattern_learner.predict_answer(question, structure)
            return pattern_answer, pattern_conf
        
        # 2. 각 답변 후보 점수 계산
        answer_scores = {}
        
        for answer in extracted_answers:
            score = 0.0
            
            # 모델 신뢰도
            score += confidence * 0.4
            
            # 위치 점수 (뒤에 나올수록 높음)
            position = model_response.rfind(answer)
            position_score = position / len(model_response) if model_response else 0
            score += position_score * 0.2
            
            # 패턴 매칭 점수
            pattern_answer, pattern_conf = self.pattern_learner.predict_answer(question, structure)
            if answer == pattern_answer:
                score += pattern_conf * 0.3
            
            # 명시적 표현 보너스
            if self._has_explicit_answer_phrase(model_response, answer):
                score += 0.1
            
            answer_scores[answer] = score
        
        # 3. 최고 점수 답변 선택
        if answer_scores:
            best_answer = max(answer_scores.items(), key=lambda x: x[1])
            
            # 신뢰도 부스트
            final_confidence = min(confidence + self.pattern_learner.get_confidence_boost(
                question, best_answer[0]
            ), 1.0)
            
            return best_answer[0], final_confidence
        
        # 폴백
        return "2", 0.3
    
    def _extract_all_possible_answers(self, response: str) -> List[str]:
        """모든 가능한 답변 추출"""
        answers = []
        
        # 다양한 패턴으로 추출
        patterns = [
            r'정답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'([1-5])번',
            r'선택지\s*([1-5])',
            r'따라서\s*([1-5])',
            r'결론.*?([1-5])'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            answers.extend(matches)
        
        # 단순 숫자도 포함
        simple_numbers = re.findall(r'[1-5]', response)
        answers.extend(simple_numbers)
        
        # 중복 제거하되 순서 유지
        seen = set()
        unique_answers = []
        for answer in answers:
            if answer not in seen:
                seen.add(answer)
                unique_answers.append(answer)
        
        return unique_answers
    
    def _has_explicit_answer_phrase(self, response: str, answer: str) -> bool:
        """명시적 답변 표현 확인"""
        explicit_phrases = [
            f"정답.*{answer}",
            f"답.*{answer}",
            f"{answer}번.*정답",
            f"{answer}번.*맞",
            f"결론.*{answer}"
        ]
        
        for phrase in explicit_phrases:
            if re.search(phrase, response, re.IGNORECASE):
                return True
        
        return False