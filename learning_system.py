# learning_system.py
"""
통합 학습
"""

import json
import pickle
import hashlib
import numpy as np
import re
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
        
    def _initialize_korean_templates(self) -> Dict[str, List[str]]:
        """한국어 템플릿 초기화"""
        return {
            "개인정보보호": [
                "개인정보보호법 제{조}조에 따라 {내용}을 수행해야 합니다.",
                "정보주체의 {권리}를 보장하기 위해 {조치}가 필요합니다.",
                "개인정보 {단계}에서 {요구사항}을 준수해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 {요구사항}을 이행해야 합니다.",
                "전자적 장치를 통한 {거래유형}에서 {보안조치}가 필요합니다.",
                "금융회사는 {의무사항}을 준수해야 합니다."
            ],
            "정보보안": [
                "정보보호관리체계는 {구성요소}를 포함해야 합니다.",
                "위험평가를 통해 {위험요소}를 식별하고 {대응방안}을 수립해야 합니다.",
                "{보안영역}에 대한 {통제방안}을 구현해야 합니다."
            ],
            "암호화": [
                "{정보유형}은 {암호알고리즘}을 사용하여 암호화해야 합니다.",
                "키 관리는 {생명주기}에 따라 {관리방안}을 적용해야 합니다.",
                "{전송/저장} 시 {암호화방식}을 적용해야 합니다."
            ],
            "사고대응": [
                "{사고유형} 발생 시 {대응절차}를 수행해야 합니다.",
                "침해사고 대응은 {단계}별로 {조치사항}을 이행해야 합니다.",
                "복구 계획은 {목표시간}을 고려하여 {복구방안}을 수립해야 합니다."
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
        
        weird_chars = re.findall(r'[^\w\s가-힣0-9.,!?()·\-]', text)
        if len(weird_chars) > 10:
            return 0.1
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책']
        prof_count = sum(1 for term in professional_terms if term in text)
        prof_bonus = min(prof_count * 0.05, 0.2)
        
        quality = korean_ratio * 0.7
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
        self.learning_metrics["total_samples"] += 1
        
        if is_correct:
            self.learning_metrics["correct_predictions"] += 1
        
        self._update_korean_quality_stats(korean_quality)
        
        if korean_quality > 0.8 and question_type != "multiple_choice":
            for d in domain:
                self.successful_answers[d].append({
                    "answer": predicted_answer,
                    "confidence": confidence,
                    "structure": self._analyze_answer_structure(predicted_answer)
                })
                if len(self.successful_answers[d]) > 50:
                    self.successful_answers[d] = self.successful_answers[d][-50:]
        
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
        
        keywords = self._extract_keywords(question_lower)
        for keyword in keywords:
            self.pattern_bank[keyword].append({
                "answer": sample.correct_answer,
                "confidence": sample.confidence if sample.is_correct else 0.0,
                "domain": sample.domain,
                "korean_quality": sample.korean_quality
            })
        
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
        
        if "법" in text:
            laws = re.findall(r'\w+법', text)
            keywords.extend(laws)
        
        articles = re.findall(r'제\d+조', text)
        keywords.extend(articles)
        
        security_terms = ["개인정보", "암호화", "보안", "인증", "접근", "통제", "관리", "전자금융", "침해", "유출"]
        for term in security_terms:
            if term in text:
                keywords.append(term)
        
        if "정의" in text:
            keywords.append("definition")
        if "해당하지" in text or "적절하지" in text:
            keywords.append("negative")
        if "방안" in text or "대책" in text:
            keywords.append("solution")
        
        return keywords
    
    def _learn_mc_pattern(self, sample: LearningData) -> None:
        """객관식 패턴 학습"""
        
        if "해당하지" in sample.question or "적절하지" in sample.question:
            pattern_key = "negative_mc"
            if pattern_key not in self.learned_rules:
                self.learned_rules[pattern_key] = {
                    "answers": defaultdict(_default_float),
                    "confidence": 0.5
                }
            
            self.learned_rules[pattern_key]["answers"][sample.correct_answer] += 1
            
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
                    "avg_quality": 0.0
                }
            
            if sample.is_correct and len(sample.correct_answer) > 50:
                self.learned_rules[pattern_key]["templates"].append({
                    "text": sample.correct_answer[:300],
                    "quality": sample.korean_quality,
                    "structure": self._analyze_answer_structure(sample.correct_answer)
                })
                
                if len(self.learned_rules[pattern_key]["templates"]) > 20:
                    self.learned_rules[pattern_key]["templates"].sort(
                        key=lambda x: x["quality"], reverse=True
                    )
                    self.learned_rules[pattern_key]["templates"] = self.learned_rules[pattern_key]["templates"][:20]
    
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
        
        self.prediction_cache[cache_key] = prediction
        
        return prediction
    
    def _predict_mc(self, question: str, domain: List[str]) -> Tuple[str, float]:
        """객관식 예측"""
        
        if "해당하지" in question or "적절하지" in question:
            if "negative_mc" in self.learned_rules:
                answers = self.learned_rules["negative_mc"]["answers"]
                if answers:
                    best_answer = max(answers.items(), key=lambda x: x[1])
                    confidence = min(best_answer[1] / sum(answers.values()), 0.8)
                    return best_answer[0], confidence
        
        for d in domain:
            if d in self.answer_statistics:
                stats = self.answer_statistics[d]
                if stats:
                    best_answer = max(stats.items(), key=lambda x: x[1])
                    total = sum(stats.values())
                    confidence = best_answer[1] / total
                    return best_answer[0], confidence * 0.7
        
        return "3", 0.3
    
    def _predict_subjective_korean(self, question: str, domain: List[str]) -> Tuple[str, float]:
        """주관식 예측"""
        
        for d in domain:
            if d in self.successful_answers and self.successful_answers[d]:
                best_answers = sorted(
                    self.successful_answers[d],
                    key=lambda x: x["confidence"],
                    reverse=True
                )[:3]
                
                if best_answers:
                    selected = best_answers[0]
                    return selected["answer"], selected["confidence"] * 0.8
        
        for d in domain:
            pattern_key = f"subj_{d}"
            if pattern_key in self.learned_rules:
                templates = self.learned_rules[pattern_key].get("templates", [])
                if templates:
                    best_template = max(templates, key=lambda x: x["quality"])
                    return best_template["text"], 0.6
        
        return self._generate_korean_fallback(domain), 0.4
    
    def _generate_korean_fallback(self, domains: List[str]) -> str:
        """한국어 폴백 응답 생성"""
        
        if "개인정보보호" in domains:
            return "개인정보보호법에 따라 개인정보의 수집과 이용 시 정보주체의 동의를 받아야 하며, 안전성 확보조치를 통해 개인정보를 보호해야 합니다. 개인정보 유출 시에는 지체 없이 정보주체에게 통지하고 필요한 조치를 취해야 합니다."
        elif "전자금융" in domains:
            return "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성과 신뢰성을 확보해야 합니다. 접근매체를 안전하게 관리하고, 거래내역을 통지하며, 사고 발생 시 적절한 손실 분담 원칙을 적용해야 합니다."
        elif "정보보안" in domains:
            return "정보보안 관리체계를 구축하여 조직의 정보자산을 체계적으로 보호해야 합니다. 위험평가를 통해 취약점을 식별하고, 관리적 기술적 물리적 보안대책을 구현하며, 지속적인 모니터링과 개선을 수행해야 합니다."
        elif "암호화" in domains:
            return "중요 정보는 안전한 암호 알고리즘을 사용하여 암호화해야 합니다. 대칭키와 공개키 암호화를 적절히 활용하고, 안전한 키 관리 체계를 구축하며, 전송 구간과 저장 시 모두 암호화를 적용해야 합니다."
        else:
            return "관련 법령과 규정에 따라 체계적인 보안 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다. 정기적인 점검과 모니터링을 통해 보안 수준을 향상시키고, 사고 발생 시 신속한 대응 체계를 구축해야 합니다."
    
    def get_current_accuracy(self) -> float:
        """현재 정확도"""
        if self.learning_metrics["total_samples"] == 0:
            return 0.0
        
        return self.learning_metrics["correct_predictions"] / self.learning_metrics["total_samples"]
    
    def optimize_rules(self) -> None:
        """규칙 최적화"""
        
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
            "training_data": [asdict(d) for d in self.training_data[-1000:]],
            "pattern_bank": dict(self.pattern_bank),
            "answer_statistics": {k: dict(v) for k, v in self.answer_statistics.items()},
            "learned_rules": self.learned_rules,
            "rule_performance": {k: list(v) for k, v in self.rule_performance.items()},
            "learning_metrics": self.learning_metrics,
            "korean_quality_stats": self.korean_quality_stats,
            "successful_answers": {k: list(v) for k, v in self.successful_answers.items()}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_learning_data(self, filepath: str = "./learning_data.pkl") -> bool:
        """학습 데이터 로드"""
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
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
        print(f"학습 시스템: {self.learning_metrics['total_samples']}개 샘플 학습")
        if self.korean_quality_stats["total_samples"] > 0:
            print(f"한국어 품질 평균: {self.korean_quality_stats['avg_quality']:.2f}")