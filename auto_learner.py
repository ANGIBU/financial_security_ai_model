# auto_learner.py
"""
자동 학습
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

class AutoLearner:
    """자동 학습 엔진"""
    
    def __init__(self):
        # 학습 파라미터
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.min_samples = 5
        
        # 패턴 저장소
        self.pattern_weights = defaultdict(lambda: defaultdict(float))
        self.pattern_counts = defaultdict(int)
        
        # 답변 분포
        self.answer_distribution = {
            "mc": defaultdict(int),
            "domain": defaultdict(lambda: defaultdict(int))
        }
        
        # 학습 히스토리
        self.learning_history = []
        self.performance_metrics = []
        
    def learn_from_prediction(self, question: str, prediction: str,
                            confidence: float, question_type: str,
                            domain: List[str]) -> None:
        """예측으로부터 학습"""
        
        # 높은 신뢰도만 학습
        if confidence < self.confidence_threshold:
            return
        
        # 패턴 추출
        patterns = self._extract_patterns(question)
        
        # 가중치 업데이트
        for pattern in patterns:
            self.pattern_weights[pattern][prediction] += confidence * self.learning_rate
            self.pattern_counts[pattern] += 1
        
        # 답변 분포 업데이트
        if question_type == "multiple_choice":
            self.answer_distribution["mc"][prediction] += 1
        
        for d in domain:
            self.answer_distribution["domain"][d][prediction] += 1
        
        # 히스토리 저장
        self.learning_history.append({
            "question_sample": question[:100],
            "prediction": prediction,
            "confidence": confidence,
            "patterns": len(patterns)
        })
    
    def _extract_patterns(self, question: str) -> List[str]:
        """패턴 추출"""
        
        patterns = []
        question_lower = question.lower()
        
        # 구조 패턴
        if "해당하지" in question_lower:
            patterns.append("negative_question")
        if "정의" in question_lower:
            patterns.append("definition_question")
        if "법" in question_lower and "따르면" in question_lower:
            patterns.append("law_based")
        
        # 도메인 패턴
        domains = {
            "personal_info": ["개인정보", "정보주체", "동의"],
            "electronic": ["전자금융", "전자적", "거래"],
            "security": ["보안", "암호화", "접근통제"]
        }
        
        for domain, keywords in domains.items():
            if any(kw in question_lower for kw in keywords):
                patterns.append(f"domain_{domain}")
        
        # 길이 패턴
        if len(question) < 200:
            patterns.append("short_question")
        elif len(question) > 500:
            patterns.append("long_question")
        else:
            patterns.append("medium_question")
        
        return patterns
    
    def predict_with_patterns(self, question: str, 
                            question_type: str) -> Tuple[str, float]:
        """패턴 기반 예측"""
        
        patterns = self._extract_patterns(question)
        
        if not patterns:
            return self._get_default_answer(question_type), 0.3
        
        # 패턴별 답변 집계
        answer_scores = defaultdict(float)
        total_weight = 0
        
        for pattern in patterns:
            if pattern in self.pattern_weights:
                pattern_weight = self.pattern_counts[pattern]
                
                for answer, weight in self.pattern_weights[pattern].items():
                    answer_scores[answer] += weight * pattern_weight
                    total_weight += pattern_weight
        
        if not answer_scores:
            return self._get_default_answer(question_type), 0.3
        
        # 최고 점수 답변
        best_answer = max(answer_scores.items(), key=lambda x: x[1])
        confidence = min(best_answer[1] / max(total_weight, 1), 0.9)
        
        return best_answer[0], confidence
    
    def _get_default_answer(self, question_type: str) -> str:
        """기본 답변"""
        
        if question_type == "multiple_choice":
            # 분포 기반 선택
            if self.answer_distribution["mc"]:
                return max(self.answer_distribution["mc"].items(), 
                          key=lambda x: x[1])[0]
            return "2"
        else:
            return "관련 규정에 따른 적절한 조치가 필요합니다."
    
    def optimize_patterns(self) -> Dict:
        """패턴 최적화"""
        
        optimized = 0
        removed = 0
        
        # 샘플이 적은 패턴 제거
        patterns_to_remove = []
        for pattern, count in self.pattern_counts.items():
            if count < self.min_samples:
                patterns_to_remove.append(pattern)
        
        for pattern in patterns_to_remove:
            del self.pattern_weights[pattern]
            del self.pattern_counts[pattern]
            removed += 1
        
        # 가중치 정규화
        for pattern in self.pattern_weights:
            total = sum(self.pattern_weights[pattern].values())
            if total > 0:
                for answer in self.pattern_weights[pattern]:
                    self.pattern_weights[pattern][answer] /= total
                optimized += 1
        
        return {
            "optimized": optimized,
            "removed": removed,
            "remaining": len(self.pattern_weights)
        }
    
    def analyze_learning_progress(self) -> Dict:
        """학습 진행 분석"""
        
        if not self.learning_history:
            return {"status": "학습 데이터 없음"}
        
        recent_history = self.learning_history[-100:]
        
        # 신뢰도 추세
        confidences = [h["confidence"] for h in recent_history]
        confidence_trend = np.mean(confidences) if confidences else 0
        
        # 패턴 다양성
        pattern_diversity = len(self.pattern_weights)
        
        # 답변 분포
        mc_distribution = dict(self.answer_distribution["mc"])
        
        return {
            "total_samples": len(self.learning_history),
            "confidence_trend": confidence_trend,
            "pattern_diversity": pattern_diversity,
            "mc_distribution": mc_distribution,
            "active_patterns": list(self.pattern_weights.keys())[:10]
        }
    
    def save_model(self, filepath: str = "./auto_learner_model.pkl") -> None:
        """모델 저장"""
        
        model_data = {
            "pattern_weights": dict(self.pattern_weights),
            "pattern_counts": dict(self.pattern_counts),
            "answer_distribution": dict(self.answer_distribution),
            "learning_history": self.learning_history[-1000:],  # 최근 1000개만
            "parameters": {
                "learning_rate": self.learning_rate,
                "confidence_threshold": self.confidence_threshold,
                "min_samples": self.min_samples
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str = "./auto_learner_model.pkl") -> bool:
        """모델 로드"""
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pattern_weights = defaultdict(lambda: defaultdict(float), 
                                             model_data["pattern_weights"])
            self.pattern_counts = defaultdict(int, model_data["pattern_counts"])
            self.answer_distribution = model_data["answer_distribution"]
            self.learning_history = model_data["learning_history"]
            
            params = model_data.get("parameters", {})
            self.learning_rate = params.get("learning_rate", 0.1)
            self.confidence_threshold = params.get("confidence_threshold", 0.7)
            self.min_samples = params.get("min_samples", 5)
            
            return True
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        print(f"자동 학습: {len(self.pattern_weights)}개 패턴 학습")