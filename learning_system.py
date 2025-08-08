# learning_system.py
"""
통합 학습
"""

import json
import pickle
import hashlib
import numpy as np
import re
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

def _default_list():
    """기본 리스트 반환"""
    return []

def _default_int():
    """기본 정수값 반환"""
    return 0

def _default_float():
    """기본 실수값 반환"""
    return 0.0

def _default_int_dict():
    """기본 정수 딕셔너리 반환"""
    return defaultdict(_default_int)

def atomic_save(obj, filepath: str) -> bool:
    """원자적 파일 저장"""
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(dir=directory)
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, filepath)
            return True
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    except Exception:
        return False

def atomic_load(filepath: str):
    """안전한 파일 로드"""
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

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
    korean_quality: float = 0.0

class UnifiedLearningSystem:
    """통합 학습 시스템"""
    
    def __init__(self):
        self.training_data = []
        self.pattern_bank = defaultdict(_default_list)
        self.answer_statistics = defaultdict(_default_int_dict)
        
        self.korean_quality_stats = {
            "total_samples": 0,
            "high_quality": 0,
            "medium_quality": 0,
            "low_quality": 0,
            "avg_quality": 0.0
        }
        
        self.learned_rules = {}
        self.rule_performance = defaultdict(_default_list)
        
        self.korean_templates = self._initialize_korean_templates()
        self.successful_answers = defaultdict(_default_list)
        
        self.performance_history = []
        self.learning_metrics = {
            "total_samples": 0,
            "correct_predictions": 0,
            "patterns_learned": 0,
            "rules_created": 0,
            "korean_quality_avg": 0.0
        }
        
        self.prediction_cache = {}
        self.max_cache_size = 100
        
    def _initialize_korean_templates(self) -> Dict[str, List[str]]:
        """한국어 템플릿 초기화"""
        return {
            "개인정보보호": [
                "개인정보보호법에 따라 {action}를 수행해야 합니다.",
                "정보주체의 권리 보호를 위해 {measure}가 필요합니다.",
                "개인정보의 안전한 관리를 위해 {requirement}가 요구됩니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 {action}를 수행해야 합니다.",
                "전자적 장치를 통한 거래의 안전성 확보를 위해 {measure}가 필요합니다.",
                "접근매체 관리와 관련하여 {requirement}를 준수해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계에 따라 {action}를 구현해야 합니다.",
                "체계적인 보안 관리를 위해 {measure}가 요구됩니다.",
                "위험평가를 통해 {requirement}를 수립해야 합니다."
            ],
            "암호화": [
                "중요 정보는 {action}를 통해 보호해야 합니다.",
                "암호화 기술을 활용하여 {measure}를 확보해야 합니다.",
                "안전한 키 관리를 위해 {requirement}가 필요합니다."
            ],
            "사고대응": [
                "{event} 발생 시 {action}를 수행해야 합니다.",
                "침해사고 대응은 {phase}별로 {measure}를 이행해야 합니다.",
                "복구 계획은 {target}을 고려하여 {requirement}를 수립해야 합니다."
            ]
        }
    
    def _evaluate_korean_quality(self, text: str, question_type: str) -> float:
        """한국어 품질 평가"""
        
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return 1.0
            return 0.5
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return 0.0
        
        if re.search(r'[а-яё]', text.lower()):
            return 0.0
        
        weird_chars = re.findall(r'[^\w\s가-힣0-9.,!?()·\-]', text)
        if len(weird_chars) > 5:
            return 0.1
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책', '절차', '방안']
        prof_count = sum(1 for term in professional_terms if term in text)
        prof_bonus = min(prof_count * 0.05, 0.2)
        
        quality = korean_ratio * 0.9
        quality -= english_ratio * 0.3
        quality += prof_bonus
        quality = max(0, min(1, quality))
        
        return quality
    
    def add_training_sample(self, question: str, correct_answer: str, 
                          predicted_answer: str, confidence: float,
                          question_type: str, domain: List[str],
                          question_id: str = None) -> None:
        """학습 샘플 추가"""
        
        is_correct = (correct_answer == predicted_answer)
        korean_quality = self._evaluate_korean_quality(predicted_answer, question_type)
        
        if korean_quality < 0.3 and question_type != "multiple_choice":
            return
        
        sample = LearningData(
            question_id=question_id or hashlib.md5(question.encode()).hexdigest()[:8],
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted_answer,
            confidence=confidence,
            is_correct=is_correct,
            question_type=question_type,
            domain=domain,
            timestamp=time.time(),
            korean_quality=korean_quality
        )
        
        self.training_data.append(sample)
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-1000:]
        
        self.learning_metrics["total_samples"] += 1
        
        if is_correct:
            self.learning_metrics["correct_predictions"] += 1
        
        self._update_korean_quality_stats(korean_quality)
        
        if korean_quality > 0.7 and question_type != "multiple_choice" and confidence > 0.6:
            for d in domain:
                self.successful_answers[d].append({
                    "answer": predicted_answer,
                    "confidence": confidence,
                    "structure": self._analyze_answer_structure(predicted_answer)
                })
                if len(self.successful_answers[d]) > 20:
                    self.successful_answers[d] = self.successful_answers[d][-20:]
        
        self._extract_patterns(sample)
    
    def _update_korean_quality_stats(self, quality: float) -> None:
        """한국어 품질 통계 업데이트"""
        self.korean_quality_stats["total_samples"] += 1
        
        if quality > 0.8:
            self.korean_quality_stats["high_quality"] += 1
        elif quality > 0.5:
            self.korean_quality_stats["medium_quality"] += 1
        else:
            self.korean_quality_stats["low_quality"] += 1
        
        total = self.korean_quality_stats["total_samples"]
        if total > 0:
            prev_avg = self.korean_quality_stats["avg_quality"]
            self.korean_quality_stats["avg_quality"] = (prev_avg * (total - 1) + quality) / total
            self.learning_metrics["korean_quality_avg"] = self.korean_quality_stats["avg_quality"]
    
    def _analyze_answer_structure(self, answer: str) -> Dict:
        """답변 구조 분석"""
        return {
            "has_numbering": bool(re.search(r'첫째|둘째|1\)|2\)', answer)),
            "has_law_reference": bool(re.search(r'법|규정|조항', answer)),
            "has_conclusion": bool(re.search(r'따라서|그러므로|결론', answer)),
            "has_examples": bool(re.search(r'예를 들어|예시|구체적으로', answer)),
            "sentence_count": len(re.split(r'[.!?]', answer)),
            "length": len(answer)
        }
    
    def _extract_patterns(self, sample: LearningData) -> None:
        """패턴 추출"""
        
        question_lower = sample.question.lower()
        
        keywords = self._extract_keywords(question_lower)[:5]
        for keyword in keywords:
            self.pattern_bank[keyword].append({
                "answer": sample.correct_answer,
                "confidence": sample.confidence if sample.is_correct else sample.confidence * 0.5,
                "domain": sample.domain,
                "korean_quality": sample.korean_quality
            })
            
            if len(self.pattern_bank[keyword]) > 50:
                self.pattern_bank[keyword] = sorted(
                    self.pattern_bank[keyword], 
                    key=lambda x: x["confidence"], 
                    reverse=True
                )[:50]
        
        for domain in sample.domain:
            if domain not in self.answer_statistics:
                self.answer_statistics[domain] = defaultdict(_default_int)
            self.answer_statistics[domain][sample.correct_answer] += 1
        
        if sample.question_type == "multiple_choice":
            self._learn_mc_pattern(sample)
        else:
            self._learn_subjective_pattern(sample)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        
        keywords = []
        
        key_terms = ["개인정보", "전자금융", "암호화", "보안", "관리체계", "위험관리", "재해복구"]
        for term in key_terms:
            if term in text:
                keywords.append(term)
        
        if "해당하지" in text or "적절하지" in text or "옳지않" in text:
            keywords.append("부정형")
        if "정의" in text or "의미" in text:
            keywords.append("정의문제")
        if "방안" in text or "대책" in text:
            keywords.append("방안문제")
        
        return keywords
    
    def _learn_mc_pattern(self, sample: LearningData) -> None:
        """객관식 패턴 학습"""
        
        if "해당하지" in sample.question or "적절하지" in sample.question:
            pattern_key = "negative_mc"
            if pattern_key not in self.learned_rules:
                self.learned_rules[pattern_key] = {
                    "answers": defaultdict(_default_float),
                    "confidence": 0.5,
                    "success_rate": 0.0,
                    "sample_count": 0
                }
            
            rule = self.learned_rules[pattern_key]
            rule["answers"][sample.correct_answer] += 1
            rule["sample_count"] += 1
            
            if sample.is_correct:
                rule["success_rate"] = (rule["success_rate"] * (rule["sample_count"] - 1) + 1) / rule["sample_count"]
            else:
                rule["success_rate"] = (rule["success_rate"] * (rule["sample_count"] - 1)) / rule["sample_count"]
            
            self.rule_performance[pattern_key].append(sample.is_correct)
    
    def _learn_subjective_pattern(self, sample: LearningData) -> None:
        """주관식 패턴 학습"""
        
        if sample.korean_quality < 0.7:
            return
        
        for domain in sample.domain:
            pattern_key = f"subj_{domain}"
            if pattern_key not in self.learned_rules:
                self.learned_rules[pattern_key] = {
                    "templates": [],
                    "keywords": [],
                    "avg_quality": 0.0,
                    "success_rate": 0.0,
                    "sample_count": 0
                }
            
            rule = self.learned_rules[pattern_key]
            
            if sample.is_correct and len(sample.correct_answer) > 50:
                rule["templates"].append({
                    "text": sample.correct_answer[:200],
                    "quality": sample.korean_quality,
                    "structure": self._analyze_answer_structure(sample.correct_answer)
                })
                
                if len(rule["templates"]) > 10:
                    rule["templates"].sort(key=lambda x: x["quality"], reverse=True)
                    rule["templates"] = rule["templates"][:10]
                
                rule["sample_count"] += 1
                if sample.is_correct:
                    rule["success_rate"] = (rule["success_rate"] * (rule["sample_count"] - 1) + 1) / rule["sample_count"]
                else:
                    rule["success_rate"] = (rule["success_rate"] * (rule["sample_count"] - 1)) / rule["sample_count"]
    
    def predict_with_learning(self, question: str, question_type: str,
                            domain: List[str]) -> Tuple[str, float]:
        """학습 기반 예측"""
        
        cache_key = hashlib.md5(f"{question}{question_type}".encode()).hexdigest()[:12]
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        if question_type == "multiple_choice":
            prediction = self._predict_mc(question, domain)
        else:
            prediction = self._predict_subjective_korean(question, domain)
        
        if len(self.prediction_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[cache_key] = prediction
        
        return prediction
    
    def _predict_mc(self, question: str, domain: List[str]) -> Tuple[str, float]:
        """객관식 예측"""
        
        if "해당하지" in question or "적절하지" in question:
            if "negative_mc" in self.learned_rules:
                rule = self.learned_rules["negative_mc"]
                if rule["sample_count"] >= 3 and rule["success_rate"] > 0.4:
                    answers = rule["answers"]
                    if answers:
                        best_answer = max(answers.items(), key=lambda x: x[1])
                        confidence = min(best_answer[1] / sum(answers.values()) * rule["success_rate"], 0.8)
                        return best_answer[0], confidence
        
        for d in domain:
            if d in self.answer_statistics:
                stats = self.answer_statistics[d]
                if stats:
                    total = sum(stats.values())
                    if total >= 3:
                        best_answer = max(stats.items(), key=lambda x: x[1])
                        confidence = best_answer[1] / total * 0.6
                        return best_answer[0], confidence
        
        return "3", 0.2
    
    def _predict_subjective_korean(self, question: str, domain: List[str]) -> Tuple[str, float]:
        """주관식 예측"""
        
        for d in domain:
            if d in self.successful_answers and self.successful_answers[d]:
                candidates = sorted(
                    self.successful_answers[d],
                    key=lambda x: x["confidence"],
                    reverse=True
                )[:2]
                
                if candidates:
                    selected = candidates[0]
                    return selected["answer"], selected["confidence"] * 0.7
        
        for d in domain:
            pattern_key = f"subj_{d}"
            if pattern_key in self.learned_rules:
                rule = self.learned_rules[pattern_key]
                if rule.get("sample_count", 0) >= 2 and rule.get("success_rate", 0) > 0.5:
                    templates = rule.get("templates", [])
                    if templates:
                        best_template = max(templates, key=lambda x: x["quality"])
                        return best_template["text"], 0.5
        
        return self._generate_korean_fallback(domain), 0.3
    
    def _generate_korean_fallback(self, domains: List[str]) -> str:
        """한국어 폴백 응답 생성"""
        
        if "개인정보보호" in domains:
            return "개인정보보호법에 따라 개인정보의 수집과 이용 시 정보주체의 동의를 받아야 하며, 안전성 확보조치를 통해 개인정보를 보호해야 합니다."
        elif "전자금융" in domains:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성과 신뢰성을 확보해야 합니다."
        elif "정보보안" in domains:
            return "정보보안 관리체계를 구축하여 조직의 정보자산을 체계적으로 보호해야 합니다."
        elif "암호화" in domains:
            return "중요 정보는 안전한 암호 알고리즘을 사용하여 암호화해야 합니다."
        else:
            return "관련 법령과 규정에 따라 체계적인 보안 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다."
    
    def get_current_accuracy(self) -> float:
        """현재 정확도"""
        if self.learning_metrics["total_samples"] == 0:
            return 0.0
        
        return self.learning_metrics["correct_predictions"] / self.learning_metrics["total_samples"]
    
    def optimize_rules(self) -> None:
        """규칙 최적화"""
        
        rules_to_remove = []
        
        for rule_name, performance in self.rule_performance.items():
            if len(performance) >= 5:
                accuracy = sum(performance) / len(performance)
                if accuracy < 0.3:
                    rules_to_remove.append(rule_name)
        
        for rule in rules_to_remove:
            if rule in self.learned_rules:
                del self.learned_rules[rule]
            if rule in self.rule_performance:
                del self.rule_performance[rule]
        
        self.learning_metrics["rules_created"] = len(self.learned_rules)
    
    def save_learning_data(self, filepath: str = "./learning_data.pkl") -> bool:
        """학습 데이터 저장"""
        
        save_data = {
            "training_data": [asdict(d) for d in self.training_data[-300:]],
            "pattern_bank": {k: v[:20] for k, v in self.pattern_bank.items()},
            "answer_statistics": {k: dict(v) for k, v in self.answer_statistics.items()},
            "learned_rules": self.learned_rules,
            "rule_performance": {k: list(v)[-20:] for k, v in self.rule_performance.items()},
            "learning_metrics": self.learning_metrics,
            "korean_quality_stats": self.korean_quality_stats,
            "successful_answers": {k: v[-10:] for k, v in self.successful_answers.items()}
        }
        
        return atomic_save(save_data, filepath)
    
    def load_learning_data(self, filepath: str = "./learning_data.pkl") -> bool:
        """학습 데이터 로드"""
        
        data = atomic_load(filepath)
        if data is None:
            return False
        
        try:
            self.training_data = [LearningData(**d) for d in data.get("training_data", [])]
            self.pattern_bank = defaultdict(_default_list, data.get("pattern_bank", {}))
            
            answer_stats = data.get("answer_statistics", {})
            self.answer_statistics = defaultdict(_default_int_dict)
            for k, v in answer_stats.items():
                self.answer_statistics[k] = defaultdict(_default_int, v)
            
            self.learned_rules = data.get("learned_rules", {})
            
            rule_perf = data.get("rule_performance", {})
            self.rule_performance = defaultdict(_default_list)
            for k, v in rule_perf.items():
                self.rule_performance[k] = list(v)
            
            self.learning_metrics = data.get("learning_metrics", self.learning_metrics)
            self.korean_quality_stats = data.get("korean_quality_stats", self.korean_quality_stats)
            
            success_answers = data.get("successful_answers", {})
            self.successful_answers = defaultdict(_default_list)
            for k, v in success_answers.items():
                self.successful_answers[k] = list(v)
            
            return True
        except Exception:
            return False
    
    def get_learning_report(self) -> Dict:
        """학습 보고서"""
        
        return {
            "total_samples": self.learning_metrics["total_samples"],
            "accuracy": self.get_current_accuracy(),
            "patterns_learned": self.learning_metrics["patterns_learned"],
            "rules_created": self.learning_metrics["rules_created"],
            "korean_quality": {
                "average": self.korean_quality_stats["avg_quality"],
                "high_quality_rate": self.korean_quality_stats["high_quality"] / max(self.korean_quality_stats["total_samples"], 1),
                "low_quality_rate": self.korean_quality_stats["low_quality"] / max(self.korean_quality_stats["total_samples"], 1)
            },
            "domain_distribution": {
                domain: len(stats) 
                for domain, stats in self.answer_statistics.items()
            },
            "successful_templates": sum(len(answers) for answers in self.successful_answers.values()),
            "cache_size": len(self.prediction_cache)
        }
    
    def cleanup(self):
        """리소스 정리"""
        self.prediction_cache.clear()
        total = self.learning_metrics['total_samples']
        quality = self.korean_quality_stats['avg_quality']
        if total > 0:
            print(f"학습 시스템: {total}개 샘플, 품질 {quality:.2f}")