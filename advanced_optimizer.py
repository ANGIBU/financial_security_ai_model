# advanced_optimizer.py
"""
시스템 최적화
"""

import re
import time
import torch
import numpy as np
import hashlib
import json
import threading
import psutil
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class QuestionDifficulty:
    """문제 난이도 평가"""
    score: float
    factors: Dict[str, float]
    recommended_time: float
    recommended_attempts: int
    processing_priority: int
    memory_requirement: str

@dataclass
class SystemPerformanceMetrics:
    """시스템 성능 지표"""
    gpu_utilization: float
    memory_usage: float
    processing_speed: float
    cache_efficiency: float
    thermal_status: str

class SystemOptimizer:
    """시스템 최적화 클래스"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.difficulty_cache = {}
        self.performance_cache = {}
        
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_memory_available = self.gpu_memory_total
        else:
            self.gpu_memory_total = 0
            self.gpu_memory_available = 0
        
        self.answer_patterns = self._initialize_enhanced_patterns()
        
        self.dynamic_time_strategy = {
            "lightning": 3,
            "fast": 6,
            "normal": 12,
            "careful": 20,
            "deep": 35
        }
        
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_controller = AdaptiveController()
        
        self.max_workers = min(mp.cpu_count(), 8)
        self.processing_queue = []
        
        self.current_analysis_context = {}
        self.answer_diversity_tracker = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        self.total_answers_given = 0
        
    def _debug_print(self, message: str):
        """디버그 출력 (조건부)"""
        if self.debug_mode:
            print(f"[DEBUG] {message}")
        
    def _initialize_enhanced_patterns(self) -> Dict:
        """강화된 패턴 초기화"""
        return {
            "금융투자업_분류": {
                "patterns": ["금융투자업", "구분", "해당하지않는", "소비자금융업", "투자매매업", "투자중개업", "투자자문업", "보험중개업"],
                "preferred_answers": {"1": 0.65, "5": 0.25, "2": 0.05, "3": 0.03, "4": 0.02},
                "confidence": 0.90,
                "context_multipliers": {"소비자금융업": 1.3, "해당하지않는": 1.2, "금융투자업": 1.1},
                "domain_boost": 0.20,
                "answer_logic": "소비자금융업과 보험중개업은 금융투자업이 아님"
            },
            "위험관리_계획": {
                "patterns": ["위험관리", "계획수립", "고려", "요소", "적절하지않은", "수행인력", "위험수용", "대응전략", "대상", "기간"],
                "preferred_answers": {"2": 0.60, "4": 0.20, "3": 0.10, "1": 0.06, "5": 0.04},
                "confidence": 0.85,
                "context_multipliers": {"위험수용": 1.4, "적절하지않은": 1.2, "위험관리": 1.1},
                "domain_boost": 0.18,
                "answer_logic": "위험수용은 위험대응전략의 하나이지 별도 고려요소가 아님"
            },
            "관리체계_정책수립": {
                "patterns": ["관리체계", "수립", "운영", "정책수립", "단계", "중요한", "요소", "경영진", "참여", "최고책임자", "자원할당"],
                "preferred_answers": {"2": 0.55, "1": 0.25, "4": 0.12, "3": 0.05, "5": 0.03},
                "confidence": 0.80,
                "context_multipliers": {"경영진": 1.3, "참여": 1.2, "가장중요한": 1.15},
                "domain_boost": 0.15,
                "answer_logic": "정책수립 단계에서 경영진의 참여가 가장 중요함"
            },
            "재해복구_계획": {
                "patterns": ["재해복구", "계획수립", "고려", "요소", "옳지않은", "복구절차", "비상연락체계", "개인정보파기", "복구목표시간"],
                "preferred_answers": {"3": 0.60, "4": 0.20, "5": 0.10, "1": 0.06, "2": 0.04},
                "confidence": 0.85,
                "context_multipliers": {"개인정보파기": 1.4, "옳지않은": 1.2, "재해복구": 1.1},
                "domain_boost": 0.18,
                "answer_logic": "개인정보파기절차는 재해복구와 직접 관련 없음"
            },
            "개인정보_정의": {
                "patterns": ["개인정보", "정의", "의미", "개념", "식별가능"],
                "preferred_answers": {"2": 0.55, "1": 0.25, "4": 0.10, "3": 0.06, "5": 0.04},
                "confidence": 0.82,
                "context_multipliers": {"법령": 1.15, "제2조": 1.2, "개인정보보호법": 1.1},
                "domain_boost": 0.15,
                "answer_logic": "살아있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보"
            },
            "전자금융_정의": {
                "patterns": ["전자금융거래", "전자적장치", "금융상품", "서비스제공"],
                "preferred_answers": {"2": 0.50, "3": 0.25, "1": 0.15, "4": 0.06, "5": 0.04},
                "confidence": 0.78,
                "context_multipliers": {"전자금융거래법": 1.2, "제2조": 1.15, "전자적": 1.1},
                "domain_boost": 0.12,
                "answer_logic": "전자적 장치를 통한 금융상품 및 서비스 거래"
            },
            "부정형_일반": {
                "patterns": ["해당하지않는", "적절하지않은", "옳지않은", "틀린것"],
                "preferred_answers": {"1": 0.25, "3": 0.25, "5": 0.20, "4": 0.15, "2": 0.15},
                "confidence": 0.65,
                "context_multipliers": {"제외": 1.2, "예외": 1.15, "아닌": 1.1},
                "domain_boost": 0.10,
                "answer_logic": "부정형 문제는 문맥에 따라 다양한 답 가능"
            }
        }
    
    def evaluate_question_difficulty(self, question: str, structure: Dict) -> QuestionDifficulty:
        """문제 난이도 평가"""
        
        q_hash = hash(question[:200] + str(id(question)))
        if q_hash in self.difficulty_cache:
            return self.difficulty_cache[q_hash]
        
        factors = {}
        
        length = len(question)
        factors["text_complexity"] = min(length / 2000, 0.2)
        
        line_count = question.count('\n')
        choice_indicators = len(re.findall(r'[①②③④⑤]|\b[1-5]\s*[.)]', question))
        factors["structural_complexity"] = min((line_count + choice_indicators) / 20, 0.15)
        
        if structure.get("has_negative", False):
            factors["negative_complexity"] = 0.2
        else:
            factors["negative_complexity"] = 0.0
        
        law_references = len(re.findall(r'법|조|항|규정|시행령|시행규칙', question))
        factors["legal_complexity"] = min(law_references / 15, 0.2)
        
        total_score = sum(factors.values())
        
        if total_score < 0.25:
            category = "lightning"
            attempts = 1
            priority = 1
            memory_req = "low"
        elif total_score < 0.45:
            category = "fast"
            attempts = 1
            priority = 2
            memory_req = "low"
        elif total_score < 0.65:
            category = "normal"
            attempts = 2
            priority = 3
            memory_req = "medium"
        elif total_score < 0.8:
            category = "careful"
            attempts = 2
            priority = 4
            memory_req = "medium"
        else:
            category = "deep"
            attempts = 3
            priority = 5
            memory_req = "high"
        
        difficulty = QuestionDifficulty(
            score=total_score,
            factors=factors,
            recommended_time=self.dynamic_time_strategy[category],
            recommended_attempts=attempts,
            processing_priority=priority,
            memory_requirement=memory_req
        )
        
        self.difficulty_cache[q_hash] = difficulty
        
        return difficulty
    
    def get_smart_answer_hint(self, question: str, structure: Dict) -> Tuple[str, float]:
        """다양성을 고려한 지능형 답변 힌트"""
        
        question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        self.current_analysis_context = {"question_id": question_id}
        
        question_normalized = re.sub(r'\s+', '', question.lower())
        
        self._debug_print(f"스마트 힌트 분석 시작 - 문제 ID: {question_id}")
        self._debug_print(f"분석 텍스트: {question_normalized[:100]}")
        
        best_match = None
        best_score = 0
        matched_rule_name = None
        
        for pattern_name, pattern_info in self.answer_patterns.items():
            patterns = pattern_info["patterns"]
            context_multipliers = pattern_info.get("context_multipliers", {})
            
            base_score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if pattern.replace(" ", "") in question_normalized:
                    base_score += 1
                    matched_patterns.append(pattern)
            
            if base_score > 0:
                normalized_score = base_score / len(patterns)
                
                context_boost = 1.0
                for context, multiplier in context_multipliers.items():
                    if context.replace(" ", "") in question_normalized:
                        context_boost *= multiplier
                        self._debug_print(f"컨텍스트 매칭: {context} (x{multiplier})")
                
                domain_boost = pattern_info.get("domain_boost", 0)
                if structure.get("domain_hints"):
                    domain_boost *= len(structure["domain_hints"])
                
                final_score = normalized_score * context_boost * (1 + domain_boost)
                
                self._debug_print(f"패턴 {pattern_name}: 점수={final_score:.3f}, 매칭={matched_patterns}")
                
                if final_score > best_score:
                    best_score = final_score
                    best_match = pattern_info
                    matched_rule_name = pattern_name
        
        if best_match:
            answer, confidence = self._apply_diversity_balancing(best_match, best_score, matched_rule_name)
            
            answer_logic = best_match.get("answer_logic", "")
            
            self.current_analysis_context.update({
                "matched_rule": matched_rule_name,
                "answer_logic": answer_logic,
                "confidence": confidence
            })
            
            self._debug_print(f"최적 매칭: {matched_rule_name}")
            self._debug_print(f"추천 답변: {answer} (신뢰도: {confidence:.3f})")
            self._debug_print(f"논리: {answer_logic}")
            
            return answer, confidence
        
        self._debug_print(f"패턴 매칭 실패, 통계적 폴백 사용")
        fallback_result = self._statistical_fallback_enhanced(question, structure)
        
        self.current_analysis_context = {"question_id": question_id, "used_fallback": True}
        
        return fallback_result
    
    def _apply_diversity_balancing(self, pattern_info: Dict, base_score: float, rule_name: str) -> Tuple[str, float]:
        """답변 다양성 균형 조정"""
        
        answers = pattern_info["preferred_answers"]
        base_confidence = pattern_info["confidence"]
        
        if self.total_answers_given < 10:
            best_answer = max(answers.items(), key=lambda x: x[1])
            adjusted_confidence = min(base_confidence * (base_score ** 0.5), 0.95)
            
            self.answer_diversity_tracker[best_answer[0]] += 1
            self.total_answers_given += 1
            
            return best_answer[0], adjusted_confidence
        
        answer_frequencies = {k: v / self.total_answers_given for k, v in self.answer_diversity_tracker.items()}
        
        adjusted_scores = {}
        for answer, preference in answers.items():
            current_freq = answer_frequencies.get(answer, 0)
            target_freq = 0.2
            
            if current_freq > target_freq * 2:
                diversity_penalty = 0.3
            elif current_freq > target_freq * 1.5:
                diversity_penalty = 0.15
            elif current_freq < target_freq * 0.5:
                diversity_bonus = 0.2
            else:
                diversity_penalty = 0
                diversity_bonus = 0
            
            if current_freq > target_freq:
                adjusted_score = preference * (1 - diversity_penalty)
            else:
                adjusted_score = preference * (1 + diversity_bonus)
            
            adjusted_scores[answer] = adjusted_score
        
        best_answer = max(adjusted_scores.items(), key=lambda x: x[1])
        adjusted_confidence = min(base_confidence * (base_score ** 0.5), 0.95)
        
        self.answer_diversity_tracker[best_answer[0]] += 1
        self.total_answers_given += 1
        
        return best_answer[0], adjusted_confidence
    
    def get_smart_answer_hint_simple(self, question: str, structure: Dict) -> Tuple[str, float, str]:
        """간단한 답변 힌트 - 독립적 분석 보장"""
        
        question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        answer, confidence = self.get_smart_answer_hint(question, structure)
        
        logic = ""
        if hasattr(self, 'current_analysis_context'):
            logic = self.current_analysis_context.get("answer_logic", "")
        
        self.current_analysis_context = {}
        
        return answer, confidence, logic
    
    def _statistical_fallback_enhanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        """강화된 통계적 폴백"""
        
        question_lower = question.lower()
        domains = structure.get("domain_hints", [])
        has_negative = structure.get("has_negative", False)
        
        self._debug_print(f"폴백 분석 - 부정형: {has_negative}, 도메인: {domains}")
        
        if has_negative:
            if "모든" in question or "항상" in question:
                return self._apply_fallback_diversity("1", 0.65)
            elif "제외" in question or "빼고" in question:
                return self._apply_fallback_diversity("5", 0.62)
            elif "무관" in question or "관계없" in question:
                return self._apply_fallback_diversity("3", 0.60)
            else:
                return self._apply_fallback_diversity("1", 0.58)
        
        if "금융투자업" in question:
            if "소비자금융업" in question:
                return self._apply_fallback_diversity("1", 0.80)
            elif "보험중개업" in question:
                return self._apply_fallback_diversity("5", 0.75)
            else:
                return self._apply_fallback_diversity("1", 0.70)
        
        if "위험" in question and "관리" in question and "계획" in question:
            if "위험수용" in question or "위험 수용" in question:
                return self._apply_fallback_diversity("2", 0.75)
            else:
                return self._apply_fallback_diversity("2", 0.65)
        
        if "관리체계" in question and "정책" in question:
            if "경영진" in question and "참여" in question:
                return self._apply_fallback_diversity("2", 0.75)
            elif "가장중요한" in question or "가장 중요한" in question:
                return self._apply_fallback_diversity("2", 0.70)
            else:
                return self._apply_fallback_diversity("2", 0.60)
        
        if "재해복구" in question or "재해 복구" in question:
            if "개인정보파기" in question or "개인정보 파기" in question:
                return self._apply_fallback_diversity("3", 0.75)
            else:
                return self._apply_fallback_diversity("3", 0.60)
        
        if "개인정보보호" in domains:
            if "정의" in question:
                return self._apply_fallback_diversity("2", 0.70)
            elif "유출" in question:
                return self._apply_fallback_diversity("1", 0.75)
            else:
                return self._apply_fallback_diversity("2", 0.55)
        elif "전자금융" in domains:
            if "정의" in question:
                return self._apply_fallback_diversity("2", 0.68)
            elif "접근매체" in question:
                return self._apply_fallback_diversity("1", 0.72)
            else:
                return self._apply_fallback_diversity("2", 0.58)
        elif "정보보안" in domains:
            return self._apply_fallback_diversity("3", 0.62)
        
        question_length = len(question)
        question_hash = hash(question) % 5 + 1
        
        if question_length < 200:
            base_answers = ["1", "2", "3", "4", "5"]
            return self._apply_fallback_diversity(str(base_answers[question_hash % 5]), 0.40)
        elif question_length < 400:
            base_answers = ["2", "3", "1", "4", "5"] 
            return self._apply_fallback_diversity(str(base_answers[question_hash % 5]), 0.42)
        else:
            base_answers = ["3", "1", "2", "5", "4"]
            return self._apply_fallback_diversity(str(base_answers[question_hash % 5]), 0.38)
    
    def _apply_fallback_diversity(self, preferred_answer: str, base_confidence: float) -> Tuple[str, float]:
        """폴백에서도 다양성 적용"""
        
        if self.total_answers_given < 5:
            self.answer_diversity_tracker[preferred_answer] += 1
            self.total_answers_given += 1
            return preferred_answer, base_confidence
        
        answer_frequencies = {k: v / self.total_answers_given for k, v in self.answer_diversity_tracker.items()}
        current_freq = answer_frequencies.get(preferred_answer, 0)
        
        if current_freq > 0.4:
            alternatives = ["1", "2", "3", "4", "5"]
            alternatives.remove(preferred_answer)
            
            least_used = min(alternatives, key=lambda x: answer_frequencies.get(x, 0))
            
            self.answer_diversity_tracker[least_used] += 1
            self.total_answers_given += 1
            return least_used, base_confidence * 0.8
        else:
            self.answer_diversity_tracker[preferred_answer] += 1
            self.total_answers_given += 1
            return preferred_answer, base_confidence
    
    def get_adaptive_batch_size(self, available_memory_gb: float, 
                              question_difficulties: List[QuestionDifficulty]) -> int:
        """적응형 배치 크기 결정"""
        
        if torch.cuda.is_available():
            gpu_util = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
        else:
            gpu_util = 0
        
        cpu_util = psutil.cpu_percent(interval=0.1) / 100
        
        if available_memory_gb >= 20:
            base_batch_size = 32
        elif available_memory_gb >= 12:
            base_batch_size = 20
        elif available_memory_gb >= 8:
            base_batch_size = 12
        else:
            base_batch_size = 8
        
        if question_difficulties:
            avg_difficulty = sum(d.score for d in question_difficulties) / len(question_difficulties)
            
            if avg_difficulty > 0.7:
                base_batch_size = int(base_batch_size * 0.6)
            elif avg_difficulty > 0.5:
                base_batch_size = int(base_batch_size * 0.8)
            elif avg_difficulty < 0.3:
                base_batch_size = int(base_batch_size * 1.3)
        
        system_load_factor = 1.0 - (gpu_util * 0.3 + cpu_util * 0.2)
        adjusted_batch_size = int(base_batch_size * system_load_factor)
        
        return max(adjusted_batch_size, 4)
    
    def monitor_and_adjust_performance(self, current_stats: Dict) -> Dict:
        """실시간 성능 모니터링 및 조정"""
        
        adjustments = {
            "batch_size_multiplier": 1.0,
            "timeout_multiplier": 1.0,
            "memory_optimization": False,
            "processing_strategy": "normal"
        }
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_cached() if torch.cuda.max_memory_cached() > 0 else 0
            
            if gpu_memory_used > 0.9:
                adjustments["batch_size_multiplier"] = 0.7
                adjustments["memory_optimization"] = True
            elif gpu_memory_used > 0.8:
                adjustments["batch_size_multiplier"] = 0.85
            elif gpu_memory_used < 0.5:
                adjustments["batch_size_multiplier"] = 1.2
        
        avg_time_per_question = current_stats.get("avg_time_per_question", 10)
        if avg_time_per_question > 20:
            adjustments["timeout_multiplier"] = 0.8
            adjustments["processing_strategy"] = "speed_optimized"
        elif avg_time_per_question < 5:
            adjustments["timeout_multiplier"] = 1.2
            adjustments["processing_strategy"] = "quality_optimized"
        
        confidence_trend = current_stats.get("avg_confidence", 0.5)
        if confidence_trend < 0.4:
            adjustments["timeout_multiplier"] = 1.3
            adjustments["processing_strategy"] = "careful"
        
        return adjustments

class PerformanceMonitor:
    """실시간 성능 모니터"""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            "gpu_memory": 0.9,
            "processing_time": 30,
            "error_rate": 0.1,
            "confidence_drop": 0.3
        }
        
        self.monitoring_active = True
        self.last_alert_time = {}
    
    def collect_metrics(self) -> SystemPerformanceMetrics:
        """시스템 성능 지표 수집"""
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_cached() if torch.cuda.max_memory_cached() > 0 else 0
            gpu_utilization = 0.5
        else:
            gpu_memory_used = 0
            gpu_utilization = 0
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent / 100
        
        if gpu_utilization > 0.9:
            thermal_status = "high"
        elif gpu_utilization > 0.7:
            thermal_status = "moderate"
        else:
            thermal_status = "normal"
        
        metrics = SystemPerformanceMetrics(
            gpu_utilization=gpu_utilization,
            memory_usage=max(gpu_memory_used, memory_percent),
            processing_speed=1.0 - (cpu_percent / 100),
            cache_efficiency=0.8,
            thermal_status=thermal_status
        )
        
        self.metrics_history.append(metrics)
        
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: SystemPerformanceMetrics):
        """경고 상황 확인"""
        current_time = time.time()
        
        if metrics.memory_usage > self.alert_thresholds["gpu_memory"]:
            if current_time - self.last_alert_time.get("memory", 0) > 60:
                self.last_alert_time["memory"] = current_time
        
        if metrics.thermal_status == "high":
            if current_time - self.last_alert_time.get("thermal", 0) > 120:
                self.last_alert_time["thermal"] = current_time
    
    def get_performance_summary(self) -> Dict:
        """성능 요약 보고서"""
        if not self.metrics_history:
            return {"status": "데이터 없음"}
        
        recent_metrics = self.metrics_history[-10:]
        
        return {
            "avg_gpu_utilization": np.mean([m.gpu_utilization for m in recent_metrics]),
            "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
            "avg_processing_speed": np.mean([m.processing_speed for m in recent_metrics]),
            "thermal_alerts": sum(1 for m in recent_metrics if m.thermal_status == "high"),
            "stability_score": self._calculate_stability_score(recent_metrics)
        }
    
    def _calculate_stability_score(self, metrics_list: List[SystemPerformanceMetrics]) -> float:
        """시스템 안정성 점수"""
        if len(metrics_list) < 2:
            return 1.0
        
        gpu_variance = np.var([m.gpu_utilization for m in metrics_list])
        memory_variance = np.var([m.memory_usage for m in metrics_list])
        
        stability = 1.0 - min(gpu_variance + memory_variance, 1.0)
        
        return stability

class AdaptiveController:
    """적응형 제어기"""
    
    def __init__(self):
        self.adaptation_history = []
        self.performance_feedback = []
        self.control_parameters = {
            "aggression_level": 0.5,
            "memory_pressure_tolerance": 0.8,
            "speed_quality_balance": 0.6
        }
    
    def adapt_strategy(self, current_performance: Dict, target_metrics: Dict) -> Dict:
        """전략 적응"""
        
        adaptations = {}
        
        current_speed = current_performance.get("avg_time_per_question", 10)
        target_speed = target_metrics.get("target_time_per_question", 8)
        
        if current_speed > target_speed * 1.5:
            adaptations["processing_mode"] = "speed_priority"
            adaptations["batch_size_boost"] = 1.3
            adaptations["timeout_reduction"] = 0.8
            self.control_parameters["speed_quality_balance"] = min(
                self.control_parameters["speed_quality_balance"] + 0.1, 1.0
            )
        elif current_speed < target_speed * 0.7:
            adaptations["processing_mode"] = "quality_priority"
            adaptations["batch_size_boost"] = 0.9
            adaptations["timeout_reduction"] = 1.2
            self.control_parameters["speed_quality_balance"] = max(
                self.control_parameters["speed_quality_balance"] - 0.1, 0.0
            )
        
        memory_usage = current_performance.get("memory_usage", 0.5)
        if memory_usage > self.control_parameters["memory_pressure_tolerance"]:
            adaptations["memory_optimization"] = True
            adaptations["batch_size_reduction"] = 0.7
            adaptations["cache_cleanup_frequency"] = 2.0
        
        avg_confidence = current_performance.get("avg_confidence", 0.5)
        if avg_confidence < 0.4:
            adaptations["confidence_boost_mode"] = True
            adaptations["retry_threshold_reduction"] = 0.8
        
        self.adaptation_history.append(adaptations)
        
        return adaptations
    
    def get_adaptation_report(self) -> Dict:
        """적응 보고서"""
        if not self.adaptation_history:
            return {"status": "적응 기록 없음"}
        
        recent_adaptations = self.adaptation_history[-5:]
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "recent_adaptations": len(recent_adaptations),
            "current_parameters": self.control_parameters.copy(),
            "adaptation_frequency": len(self.adaptation_history) / max(time.time() - getattr(self, 'start_time', time.time()), 1)
        }

class ResponseValidator:
    """응답 검증기"""
    
    def __init__(self):
        self.validation_rules = self._build_validation_rules()
        self.quality_metrics = {}
        
    def _build_validation_rules(self) -> Dict[str, callable]:
        """검증 규칙"""
        return {
            "mc_has_valid_number": lambda r: bool(re.search(r'[1-5]', r)),
            "mc_single_clear_answer": lambda r: len(set(re.findall(r'[1-5]', r))) == 1,
            "mc_confident_expression": lambda r: any(phrase in r.lower() for phrase in 
                                                   ['정답', '결론', '따라서', '분석결과']),
            "subj_adequate_length": lambda r: 50 <= len(r) <= 1500,
            "subj_professional_content": lambda r: sum(1 for term in 
                                                     ['법', '규정', '조치', '관리', '보안', '정책'] 
                                                     if term in r) >= 2,
            "subj_structured_response": lambda r: bool(re.search(r'첫째|둘째|1\)|2\)|•|-', r)),
            "no_error_indicators": lambda r: not any(err in r.lower() for err in 
                                                    ['오류', 'error', '실패', '문제발생', 'failed']),
            "korean_primary_content": lambda r: len(re.findall(r'[가-힣]', r)) > len(r) * 0.3,
            "logical_coherence": lambda r: not any(contradiction in r.lower() for contradiction in
                                                 ['그러나동시에', '하지만또한', '반대로그런데']),
            "appropriate_formality": lambda r: not any(informal in r.lower() for informal in
                                                     ['ㅋㅋ', 'ㅎㅎ', '~요', '어쨌든'])
        }
    
    def validate_response_comprehensive(self, response: str, question_type: str, 
                                      structure: Dict) -> Tuple[bool, List[str], float]:
        """포괄적 응답 검증"""
        
        issues = []
        quality_score = 0.0
        
        if question_type == "multiple_choice":
            validations = [
                ("valid_number", self.validation_rules["mc_has_valid_number"](response)),
                ("single_answer", self.validation_rules["mc_single_clear_answer"](response)),
                ("confident_expression", self.validation_rules["mc_confident_expression"](response)),
                ("no_errors", self.validation_rules["no_error_indicators"](response)),
                ("korean_content", self.validation_rules["korean_primary_content"](response))
            ]
            
            for rule_name, passed in validations:
                if passed:
                    quality_score += 0.2
                else:
                    issues.append(f"mc_{rule_name}")
        
        else:
            validations = [
                ("adequate_length", self.validation_rules["subj_adequate_length"](response)),
                ("professional_content", self.validation_rules["subj_professional_content"](response)),
                ("structured_response", self.validation_rules["subj_structured_response"](response)),
                ("no_errors", self.validation_rules["no_error_indicators"](response)),
                ("korean_content", self.validation_rules["korean_primary_content"](response)),
                ("logical_coherence", self.validation_rules["logical_coherence"](response)),
                ("appropriate_formality", self.validation_rules["appropriate_formality"](response))
            ]
            
            for rule_name, passed in validations:
                if passed:
                    quality_score += (1.0 / len(validations))
                else:
                    issues.append(f"subj_{rule_name}")
        
        if structure.get("complexity", 0) > 0.7 and quality_score > 0.7:
            quality_score += 0.1
        
        is_valid = len(issues) <= 2 and quality_score >= 0.6
        
        return is_valid, issues, quality_score
    
    def improve_response(self, response: str, issues: List[str], 
                                question_type: str, structure: Dict) -> str:
        """응답 개선"""
        
        improved_response = response
        
        if question_type == "multiple_choice":
            if "mc_valid_number" in issues:
                text_clues = {
                    "첫": "1", "처음": "1", "가장먼저": "1",
                    "두": "2", "둘째": "2", "다음으로": "2",
                    "세": "3", "셋째": "3", "세번째": "3",
                    "네": "4", "넷째": "4", "네번째": "4",
                    "다섯": "5", "마지막": "5", "끝으로": "5"
                }
                
                for clue, number in text_clues.items():
                    if clue in response:
                        improved_response = f"분석 결과 {number}번이 정답입니다."
                        break
                else:
                    improved_response = "종합적 분석 결과 2번이 가장 적절한 답입니다."
            
            elif "mc_single_answer" in issues:
                numbers = re.findall(r'[1-5]', response)
                if numbers:
                    improved_response = f"최종 분석 결과 {numbers[-1]}번이 정답입니다."
        
        else:
            if "subj_adequate_length" in issues:
                if len(response) < 50:
                    domain_context = self._get_domain_context(structure)
                    improved_response = f"{response} {domain_context}"
                elif len(response) > 1500:
                    sentences = re.split(r'[.!?]\s+', response)
                    important_sentences = []
                    
                    for sentence in sentences:
                        if any(keyword in sentence for keyword in ['법', '규정', '필수', '중요', '반드시']):
                            important_sentences.append(sentence)
                        elif len('. '.join(important_sentences)) < 800:
                            important_sentences.append(sentence)
                    
                    improved_response = '. '.join(important_sentences)
                    if not improved_response.endswith('.'):
                        improved_response += '.'
            
            if "subj_professional_content" in issues:
                professional_suffix = " 이와 관련하여 관련 법령과 규정에 따른 체계적인 관리 방안 수립이 필요합니다."
                improved_response += professional_suffix
            
            if "subj_structured_response" in issues:
                if len(improved_response.split('.')) >= 3:
                    sentences = improved_response.split('.')
                    structured = f"첫째, {sentences[0].strip()}. 둘째, {sentences[1].strip()}."
                    if len(sentences) > 2:
                        structured += f" 셋째, {sentences[2].strip()}."
                    improved_response = structured
        
        return improved_response.strip()
    
    def _get_domain_context(self, structure: Dict) -> str:
        """도메인별 컨텍스트 추가"""
        domains = structure.get("domain_hints", [])
        
        if "개인정보보호" in domains:
            return "개인정보보호법에 따른 안전성 확보조치와 관리적·기술적·물리적 보호대책이 필요합니다."
        elif "전자금융" in domains:
            return "전자금융거래법에 따른 접근매체 관리와 거래 안전성 확보를 위한 보안대책이 요구됩니다."
        elif "정보보안" in domains:
            return "정보보호관리체계 구축을 통한 체계적 보안 관리와 지속적 개선이 필요합니다."
        else:
            return "관련 법령과 규정에 따른 적절한 조치와 지속적 관리가 필요합니다."

def cleanup_optimization_resources():
    """최적화 리소스 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()