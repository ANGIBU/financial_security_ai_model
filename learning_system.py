# learning_system.py
"""
통합 학습 시스템
"""

import json
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

@dataclass
class LearningData:
    """학습 데이터"""
    question_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    confidence: float
    is_correct: bool
    question_type: str
    domain: List[str]
    timestamp: float

class UnifiedLearningSystem:
    """통합 학습 시스템"""
    
    def __init__(self):
        # 학습 데이터 저장소
        self.training_data = []
        self.pattern_bank = defaultdict(list)
        self.answer_statistics = defaultdict(lambda: defaultdict(int))
        
        # 학습 규칙
        self.learned_rules = {}
        self.rule_performance = defaultdict(list)
        
        # 성능 추적
        self.performance_history = []
        self.learning_metrics = {
            "total_samples": 0,
            "correct_predictions": 0,
            "patterns_learned": 0,
            "rules_created": 0
        }
        
        # 캐시
        self.prediction_cache = {}
        
    def add_training_sample(self, question: str, correct_answer: str, 
                          predicted_answer: str, confidence: float,
                          question_type: str, domain: List[str],
                          question_id: str = None) -> None:
        """학습 샘플 추가"""
        
        is_correct = (correct_answer == predicted_answer)
        
        sample = LearningData(
            question_id=question_id or hashlib.md5(question.encode()).hexdigest()[:8],
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted_answer,
            confidence=confidence,
            is_correct=is_correct,
            question_type=question_type,
            domain=domain,
            timestamp=time.time()
        )
        
        self.training_data.append(sample)
        self.learning_metrics["total_samples"] += 1
        
        if is_correct:
            self.learning_metrics["correct_predictions"] += 1
        
        # 패턴 추출 및 저장
        self._extract_patterns(sample)
        
    def _extract_patterns(self, sample: LearningData) -> None:
        """패턴 추출"""
        
        question_lower = sample.question.lower()
        
        # 키워드 기반 패턴
        keywords = self._extract_keywords(question_lower)
        for keyword in keywords:
            self.pattern_bank[keyword].append({
                "answer": sample.correct_answer,
                "confidence": sample.confidence if sample.is_correct else 0.0,
                "domain": sample.domain
            })
        
        # 도메인별 답변 분포
        for domain in sample.domain:
            self.answer_statistics[domain][sample.correct_answer] += 1
        
        # 문제 유형별 패턴
        if sample.question_type == "multiple_choice":
            self._learn_mc_pattern(sample)
        else:
            self._learn_subjective_pattern(sample)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        import re
        
        # 핵심 키워드 추출
        keywords = []
        
        # 법령 관련
        if "법" in text:
            laws = re.findall(r'\w+법', text)
            keywords.extend(laws)
        
        # 조항 관련
        articles = re.findall(r'제\d+조', text)
        keywords.extend(articles)
        
        # 보안 용어
        security_terms = ["개인정보", "암호화", "보안", "인증", "접근", "통제", "관리"]
        for term in security_terms:
            if term in text:
                keywords.append(term)
        
        return keywords
    
    def _learn_mc_pattern(self, sample: LearningData) -> None:
        """객관식 패턴 학습"""
        
        # 부정형 문제 패턴
        if "해당하지" in sample.question or "적절하지" in sample.question:
            pattern_key = "negative_mc"
            if pattern_key not in self.learned_rules:
                self.learned_rules[pattern_key] = {
                    "answers": defaultdict(float),
                    "confidence": 0.5
                }
            
            self.learned_rules[pattern_key]["answers"][sample.correct_answer] += 1
            
            # 성능 추적
            self.rule_performance[pattern_key].append(sample.is_correct)
    
    def _learn_subjective_pattern(self, sample: LearningData) -> None:
        """주관식 패턴 학습"""
        
        # 도메인별 템플릿 학습
        for domain in sample.domain:
            pattern_key = f"subj_{domain}"
            if pattern_key not in self.learned_rules:
                self.learned_rules[pattern_key] = {
                    "templates": [],
                    "keywords": []
                }
            
            if sample.is_correct and len(sample.correct_answer) > 50:
                # 성공적인 답변 템플릿 저장
                self.learned_rules[pattern_key]["templates"].append(sample.correct_answer[:200])
    
    def predict_with_learning(self, question: str, question_type: str,
                            domain: List[str]) -> Tuple[str, float]:
        """학습 기반 예측"""
        
        # 캐시 확인
        cache_key = hashlib.md5(f"{question}{question_type}".encode()).hexdigest()[:12]
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        if question_type == "multiple_choice":
            prediction = self._predict_mc(question, domain)
        else:
            prediction = self._predict_subjective(question, domain)
        
        # 캐시 저장
        self.prediction_cache[cache_key] = prediction
        
        return prediction
    
    def _predict_mc(self, question: str, domain: List[str]) -> Tuple[str, float]:
        """객관식 예측"""
        
        # 부정형 확인
        if "해당하지" in question or "적절하지" in question:
            if "negative_mc" in self.learned_rules:
                answers = self.learned_rules["negative_mc"]["answers"]
                if answers:
                    best_answer = max(answers.items(), key=lambda x: x[1])
                    confidence = min(best_answer[1] / sum(answers.values()), 0.8)
                    return best_answer[0], confidence
        
        # 도메인 기반 예측
        for d in domain:
            if d in self.answer_statistics:
                stats = self.answer_statistics[d]
                if stats:
                    best_answer = max(stats.items(), key=lambda x: x[1])
                    total = sum(stats.values())
                    confidence = best_answer[1] / total
                    return best_answer[0], confidence * 0.7
        
        # 기본값
        return "2", 0.3
    
    def _predict_subjective(self, question: str, domain: List[str]) -> Tuple[str, float]:
        """주관식 예측"""
        
        # 도메인별 템플릿 활용
        for d in domain:
            pattern_key = f"subj_{d}"
            if pattern_key in self.learned_rules:
                templates = self.learned_rules[pattern_key].get("templates", [])
                if templates:
                    # 가장 최근 템플릿 활용
                    return templates[-1], 0.5
        
        # 기본 응답
        if "개인정보" in domain:
            return "개인정보보호법에 따른 안전성 확보조치가 필요합니다.", 0.4
        elif "전자금융" in domain:
            return "전자금융거래법에 따른 보안 대책이 요구됩니다.", 0.4
        else:
            return "관련 법령에 따른 적절한 조치가 필요합니다.", 0.3
    
    def auto_learn_from_batch(self, questions: List[Dict], 
                            predictions: List[Dict]) -> Dict:
        """배치 자동 학습"""
        
        learned_patterns = 0
        
        for q_data, pred in zip(questions, predictions):
            # 예측 신뢰도가 높은 경우 학습
            if pred.get("confidence", 0) > 0.7:
                self.add_training_sample(
                    question=q_data["question"],
                    correct_answer=pred["answer"],
                    predicted_answer=pred["answer"],
                    confidence=pred["confidence"],
                    question_type=q_data.get("type", "multiple_choice"),
                    domain=q_data.get("domain", ["일반"]),
                    question_id=q_data.get("id")
                )
                learned_patterns += 1
        
        self.learning_metrics["patterns_learned"] += learned_patterns
        
        return {
            "patterns_learned": learned_patterns,
            "total_samples": len(self.training_data),
            "accuracy": self.get_current_accuracy()
        }
    
    def get_current_accuracy(self) -> float:
        """현재 정확도"""
        if self.learning_metrics["total_samples"] == 0:
            return 0.0
        
        return self.learning_metrics["correct_predictions"] / self.learning_metrics["total_samples"]
    
    def optimize_rules(self) -> None:
        """규칙 최적화"""
        
        # 성능이 낮은 규칙 제거
        rules_to_remove = []
        
        for rule_name, performance in self.rule_performance.items():
            if len(performance) >= 10:
                accuracy = sum(performance) / len(performance)
                if accuracy < 0.3:
                    rules_to_remove.append(rule_name)
        
        for rule in rules_to_remove:
            del self.learned_rules[rule]
            del self.rule_performance[rule]
        
        self.learning_metrics["rules_created"] = len(self.learned_rules)
    
    def save_learning_data(self, filepath: str = "./learning_data.pkl") -> None:
        """학습 데이터 저장"""
        
        save_data = {
            "training_data": [asdict(d) for d in self.training_data],
            "pattern_bank": dict(self.pattern_bank),
            "answer_statistics": dict(self.answer_statistics),
            "learned_rules": self.learned_rules,
            "rule_performance": dict(self.rule_performance),
            "learning_metrics": self.learning_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_learning_data(self, filepath: str = "./learning_data.pkl") -> bool:
        """학습 데이터 로드"""
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.training_data = [LearningData(**d) for d in data["training_data"]]
            self.pattern_bank = defaultdict(list, data["pattern_bank"])
            self.answer_statistics = defaultdict(lambda: defaultdict(int), data["answer_statistics"])
            self.learned_rules = data["learned_rules"]
            self.rule_performance = defaultdict(list, data["rule_performance"])
            self.learning_metrics = data["learning_metrics"]
            
            return True
        except Exception as e:
            print(f"학습 데이터 로드 실패: {e}")
            return False
    
    def get_learning_report(self) -> Dict:
        """학습 보고서"""
        
        return {
            "total_samples": self.learning_metrics["total_samples"],
            "accuracy": self.get_current_accuracy(),
            "patterns_learned": self.learning_metrics["patterns_learned"],
            "rules_created": self.learning_metrics["rules_created"],
            "domain_distribution": {
                domain: len(stats) 
                for domain, stats in self.answer_statistics.items()
            },
            "cache_size": len(self.prediction_cache)
        }
    
    def cleanup(self):
        """리소스 정리"""
        self.prediction_cache.clear()
        print(f"학습 시스템: {self.learning_metrics['total_samples']}개 샘플 학습")