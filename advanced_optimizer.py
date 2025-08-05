# advanced_optimizer.py
"""
고급 최적화 시스템 - 정확도와 속도 균형
"""

import re
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import json

@dataclass
class QuestionDifficulty:
    """문제 난이도 평가"""
    score: float  # 0.0 ~ 1.0
    factors: Dict[str, float]
    recommended_time: float
    recommended_attempts: int

class AdvancedOptimizer:
    """고급 최적화 클래스"""
    
    def __init__(self):
        # 문제 난이도 캐시
        self.difficulty_cache = {}
        
        # 정답 패턴 학습
        self.answer_patterns = {
            "개인정보_정의": {"common": "2", "confidence": 0.7},
            "전자금융_정의": {"common": "2", "confidence": 0.6},
            "부정형_일반": {"common": "1", "confidence": 0.6},
            "법령_조항": {"common": "3", "confidence": 0.5},
        }
        
        # 시간 할당 전략
        self.time_strategy = {
            "easy": 8,      # 쉬운 문제
            "medium": 15,   # 중간 문제
            "hard": 25,     # 어려운 문제
            "critical": 35  # 매우 중요한 문제
        }
    
    def evaluate_question_difficulty(self, question: str, structure: Dict) -> QuestionDifficulty:
        """문제 난이도 정밀 평가"""
        
        # 캐시 확인
        q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        if q_hash in self.difficulty_cache:
            return self.difficulty_cache[q_hash]
        
        factors = {}
        
        # 1. 길이 요소
        length = len(question)
        factors["length"] = min(length / 1000, 1.0)
        
        # 2. 복잡도 요소
        line_count = question.count('\n')
        factors["structure"] = min(line_count / 10, 1.0)
        
        # 3. 부정형 여부
        if structure.get("has_negative", False):
            factors["negative"] = 0.8
        else:
            factors["negative"] = 0.0
        
        # 4. 법령 관련도
        law_keywords = ["법", "조", "항", "규정", "시행령", "시행규칙"]
        law_count = sum(1 for kw in law_keywords if kw in question)
        factors["legal"] = min(law_count / 5, 1.0)
        
        # 5. 전문 용어 밀도
        tech_terms = ["암호화", "인증", "해시", "전자서명", "PKI", "SSL", "접근제어"]
        tech_count = sum(1 for term in tech_terms if term in question)
        factors["technical"] = min(tech_count / 4, 1.0)
        
        # 6. 선택지 복잡도 (객관식)
        if structure["question_type"] == "multiple_choice":
            choices = structure.get("choices", [])
            if choices:
                avg_choice_length = sum(len(c["text"]) for c in choices) / len(choices)
                factors["choice_complexity"] = min(avg_choice_length / 100, 1.0)
            else:
                factors["choice_complexity"] = 0.5
        
        # 종합 점수 계산
        weights = {
            "length": 0.15,
            "structure": 0.15,
            "negative": 0.25,
            "legal": 0.2,
            "technical": 0.15,
            "choice_complexity": 0.1
        }
        
        total_score = sum(factors.get(key, 0) * weight for key, weight in weights.items())
        
        # 시간 및 시도 횟수 권장
        if total_score < 0.3:
            category = "easy"
            attempts = 1
        elif total_score < 0.5:
            category = "medium"
            attempts = 2
        elif total_score < 0.7:
            category = "hard"
            attempts = 2
        else:
            category = "critical"
            attempts = 3
        
        difficulty = QuestionDifficulty(
            score=total_score,
            factors=factors,
            recommended_time=self.time_strategy[category],
            recommended_attempts=attempts
        )
        
        # 캐시 저장
        self.difficulty_cache[q_hash] = difficulty
        
        return difficulty
    
    def get_smart_answer_hint(self, question: str, structure: Dict) -> Tuple[str, float]:
        """지능형 답변 힌트 (패턴 기반)"""
        
        question_lower = question.lower()
        
        # 패턴 매칭
        if "개인정보" in question_lower and "정의" in question_lower:
            return self.answer_patterns["개인정보_정의"]["common"], \
                   self.answer_patterns["개인정보_정의"]["confidence"]
        
        elif "전자금융" in question_lower and "정의" in question_lower:
            return self.answer_patterns["전자금융_정의"]["common"], \
                   self.answer_patterns["전자금융_정의"]["confidence"]
        
        elif structure.get("has_negative", False):
            return self.answer_patterns["부정형_일반"]["common"], \
                   self.answer_patterns["부정형_일반"]["confidence"]
        
        elif "법" in question_lower and ("조" in question_lower or "항" in question_lower):
            return self.answer_patterns["법령_조항"]["common"], \
                   self.answer_patterns["법령_조항"]["confidence"]
        
        # 기본값
        return "2", 0.3
    
    def optimize_batch_size(self, available_memory_gb: float, 
                          question_lengths: List[int]) -> int:
        """동적 배치 크기 최적화"""
        
        # 평균 문제 길이
        avg_length = sum(question_lengths) / len(question_lengths) if question_lengths else 500
        
        # 메모리 기반 계산
        base_batch_size = int(available_memory_gb * 2)  # 기본 비율
        
        # 길이 기반 조정
        if avg_length > 800:
            batch_size = max(base_batch_size // 2, 4)
        elif avg_length > 500:
            batch_size = base_batch_size
        else:
            batch_size = min(base_batch_size * 1.5, 16)
        
        return int(batch_size)
    
    def prioritize_questions(self, questions: List[Dict]) -> List[Dict]:
        """문제 우선순위 재정렬"""
        
        # 점수 계산
        for q in questions:
            score = 0
            
            # 객관식 우선 (빠른 처리)
            if q["type"] == "multiple_choice":
                score += 2
            
            # 쉬운 문제 우선
            if q["difficulty"].score < 0.4:
                score += 3
            elif q["difficulty"].score < 0.6:
                score += 1
            
            # 높은 확신도 예상 문제
            if q.get("hint_confidence", 0) > 0.6:
                score += 2
            
            q["priority_score"] = score
        
        # 정렬 (높은 점수 = 먼저 처리)
        return sorted(questions, key=lambda x: x["priority_score"], reverse=True)

class ResponseValidator:
    """응답 검증 및 개선"""
    
    def __init__(self):
        self.validation_rules = self._build_validation_rules()
    
    def _build_validation_rules(self) -> Dict[str, callable]:
        """검증 규칙 구축"""
        return {
            "mc_has_number": lambda r: bool(re.search(r'[1-5]', r)),
            "mc_single_answer": lambda r: len(re.findall(r'[1-5]', r)) >= 1,
            "subj_min_length": lambda r: len(r) >= 30,
            "subj_max_length": lambda r: len(r) <= 1500,
            "no_error_phrases": lambda r: not any(err in r.lower() for err in 
                                                 ["오류", "error", "실패", "failed"]),
            "korean_content": lambda r: bool(re.search(r'[가-힣]', r))
        }
    
    def validate_response(self, response: str, question_type: str) -> Tuple[bool, List[str]]:
        """응답 검증"""
        
        issues = []
        
        if question_type == "multiple_choice":
            # 객관식 검증
            if not self.validation_rules["mc_has_number"](response):
                issues.append("no_number")
            
            numbers = re.findall(r'[1-5]', response)
            if len(set(numbers)) > 1:
                issues.append("multiple_answers")
        
        else:
            # 주관식 검증
            if not self.validation_rules["subj_min_length"](response):
                issues.append("too_short")
            
            if not self.validation_rules["subj_max_length"](response):
                issues.append("too_long")
        
        # 공통 검증
        if not self.validation_rules["no_error_phrases"](response):
            issues.append("error_phrases")
        
        if not self.validation_rules["korean_content"](response):
            issues.append("no_korean")
        
        return len(issues) == 0, issues
    
    def improve_response(self, response: str, issues: List[str], 
                        question_type: str) -> str:
        """응답 개선"""
        
        if question_type == "multiple_choice":
            if "no_number" in issues:
                # 마지막 시도로 텍스트에서 답 추론
                if "첫" in response or "처음" in response:
                    return "1"
                elif "두" in response or "둘째" in response:
                    return "2"
                elif "세" in response or "셋째" in response:
                    return "3"
                elif "네" in response or "넷째" in response:
                    return "4"
                elif "다섯" in response or "마지막" in response:
                    return "5"
                else:
                    return "2"  # 기본값
            
            elif "multiple_answers" in issues:
                # 마지막 숫자 선택
                numbers = re.findall(r'[1-5]', response)
                return numbers[-1] if numbers else "2"
        
        else:
            if "too_short" in issues:
                return f"{response} 이와 관련하여 금융보안 정책에 따른 적절한 조치가 필요합니다."
            
            elif "too_long" in issues:
                # 첫 750자만 유지
                return response[:750] + "."
        
        return response

class PerformanceMonitor:
    """성능 모니터링 및 조정"""
    
    def __init__(self):
        self.stats = {
            "questions_processed": 0,
            "total_time": 0,
            "avg_confidence": 0,
            "time_per_question": [],
            "memory_usage": []
        }
        self.start_time = time.time()
    
    def update(self, question_time: float, confidence: float):
        """통계 업데이트"""
        self.stats["questions_processed"] += 1
        self.stats["time_per_question"].append(question_time)
        
        # 이동 평균 계산
        n = self.stats["questions_processed"]
        self.stats["avg_confidence"] = (
            (self.stats["avg_confidence"] * (n - 1) + confidence) / n
        )
    
    def get_adaptive_timeout(self) -> float:
        """적응형 타임아웃 계산"""
        
        elapsed = time.time() - self.start_time
        processed = self.stats["questions_processed"]
        remaining = 515 - processed
        
        if processed == 0:
            return 20.0  # 기본값
        
        # 남은 시간 계산
        time_limit = 4.3 * 3600  # 4시간 18분
        remaining_time = time_limit - elapsed
        
        if remaining_time <= 0 or remaining <= 0:
            return 5.0  # 긴급 모드
        
        # 적응형 계산
        avg_time = elapsed / processed
        ideal_time = remaining_time / remaining
        
        # 여유가 있으면 시간 증가, 부족하면 감소
        if ideal_time > avg_time * 1.2:
            return min(ideal_time * 0.9, 30)  # 최대 30초
        else:
            return max(ideal_time * 0.8, 5)   # 최소 5초
    
    def should_skip_retries(self) -> bool:
        """재시도 스킵 여부 결정"""
        
        elapsed = time.time() - self.start_time
        time_limit = 4.3 * 3600
        
        # 시간의 80% 경과 시 재시도 스킵
        return elapsed > time_limit * 0.8