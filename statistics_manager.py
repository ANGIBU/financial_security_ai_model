# statistics_manager.py

import time
import sys
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil을 사용할 수 없습니다. 메모리 모니터링이 제한됩니다.")

from config import LOG_DIR, MEMORY_CONFIG, DOMAIN_WEIGHTS


class StatisticsManager:
    """실행 통계 및 성능 분석"""
    
    def __init__(self, log_type: str = "inference"):
        try:
            LOG_DIR.mkdir(exist_ok=True)
            if log_type == "test":
                self.log_file = LOG_DIR / "test.txt"
            else:
                self.log_file = LOG_DIR / "inference.txt"
        except Exception as e:
            print(f"로그 디렉토리 생성 실패: {e}")
            self.log_file = Path("inference.txt")
        
        self.start_time = None
        self.processing_times = []
        self.domain_stats = {}
        self.method_stats = {}
        self.error_stats = {}
        self.template_usage_stats = {}
        self.memory_snapshots = []
        self.accuracy_tracking = {}
        self.question_type_stats = {}
        self.performance_trends = []
        self.memory_monitor_enabled = PSUTIL_AVAILABLE
        self.session_started = False
        
    def start_session(self):
        """세션 시작"""
        self.start_time = time.time()
        self.session_started = True
        self._clear_log_and_write_system_info()
        
    def record_question_processing(self, processing_time: float, domain: str, method: str, 
                                  question_type: str, success: bool, error_type: str = None):
        """문항 처리 기록"""
        try:
            self.processing_times.append(processing_time)
            
            # 도메인 통계
            if domain not in self.domain_stats:
                self.domain_stats[domain] = {
                    "count": 0, 
                    "avg_time": 0, 
                    "total_time": 0, 
                    "success_count": 0,
                    "failure_count": 0,
                    "efficiency_score": 0
                }
            
            domain_stat = self.domain_stats[domain]
            domain_stat["count"] += 1
            domain_stat["total_time"] += processing_time
            domain_stat["avg_time"] = domain_stat["total_time"] / domain_stat["count"]
            
            if success:
                domain_stat["success_count"] += 1
            else:
                domain_stat["failure_count"] += 1
            
            # 도메인별 효율성 점수 계산
            success_rate = domain_stat["success_count"] / domain_stat["count"]
            time_efficiency = max(0, 1 - (domain_stat["avg_time"] / 30))  # 30초 기준
            domain_stat["efficiency_score"] = (success_rate * 0.7 + time_efficiency * 0.3) * 100
            
            # 처리 방법 통계
            if method not in self.method_stats:
                self.method_stats[method] = {
                    "count": 0, 
                    "success_count": 0, 
                    "avg_time": 0, 
                    "total_time": 0,
                    "reliability_score": 0
                }
            
            method_stat = self.method_stats[method]
            method_stat["count"] += 1
            method_stat["total_time"] += processing_time
            method_stat["avg_time"] = method_stat["total_time"] / method_stat["count"]
            
            if success:
                method_stat["success_count"] += 1
            
            # 신뢰도 점수 계산
            method_success_rate = method_stat["success_count"] / method_stat["count"]
            method_stat["reliability_score"] = method_success_rate * method_stat["count"] / 100
            
            # 질문 타입 통계
            if question_type not in self.question_type_stats:
                self.question_type_stats[question_type] = {
                    "count": 0,
                    "success_count": 0,
                    "avg_time": 0,
                    "total_time": 0
                }
            
            qt_stat = self.question_type_stats[question_type]
            qt_stat["count"] += 1
            qt_stat["total_time"] += processing_time
            qt_stat["avg_time"] = qt_stat["total_time"] / qt_stat["count"]
            
            if success:
                qt_stat["success_count"] += 1
            
            # 정확도 추적
            if domain not in self.accuracy_tracking:
                self.accuracy_tracking[domain] = {
                    "recent_results": [],
                    "trend": "stable",
                    "peak_accuracy": 0,
                    "current_streak": 0
                }
            
            acc_track = self.accuracy_tracking[domain]
            acc_track["recent_results"].append(1 if success else 0)
            
            # 최근 10개 결과만 유지
            if len(acc_track["recent_results"]) > 10:
                acc_track["recent_results"] = acc_track["recent_results"][-10:]
            
            # 현재 정확도 계산
            current_acc = sum(acc_track["recent_results"]) / len(acc_track["recent_results"]) * 100
            if current_acc > acc_track["peak_accuracy"]:
                acc_track["peak_accuracy"] = current_acc
            
            # 연속 성공 추적
            if success:
                acc_track["current_streak"] += 1
            else:
                acc_track["current_streak"] = 0
            
            # 트렌드 분석
            if len(acc_track["recent_results"]) >= 6:
                recent_half = acc_track["recent_results"][-3:]
                earlier_half = acc_track["recent_results"][-6:-3]
                recent_rate = sum(recent_half) / len(recent_half)
                earlier_rate = sum(earlier_half) / len(earlier_half)
                
                if recent_rate > earlier_rate + 0.1:
                    acc_track["trend"] = "improving"
                elif recent_rate < earlier_rate - 0.1:
                    acc_track["trend"] = "declining"
                else:
                    acc_track["trend"] = "stable"
            
            # 오류 통계
            if error_type and not success:
                if error_type not in self.error_stats:
                    self.error_stats[error_type] = {
                        "count": 0, 
                        "avg_time": 0, 
                        "total_time": 0,
                        "domain_distribution": {}
                    }
                
                error_stat = self.error_stats[error_type]
                error_stat["count"] += 1
                error_stat["total_time"] += processing_time
                error_stat["avg_time"] = error_stat["total_time"] / error_stat["count"]
                
                if domain not in error_stat["domain_distribution"]:
                    error_stat["domain_distribution"][domain] = 0
                error_stat["domain_distribution"][domain] += 1
                
                # 로그에 오류 상세 기록
                self._log_error_details(error_type, domain, method, processing_time)
            
            # 성능 트렌드 기록
            self.performance_trends.append({
                "timestamp": time.time(),
                "processing_time": processing_time,
                "success": success,
                "domain": domain,
                "method": method,
                "question_type": question_type
            })
            
            # 트렌드 데이터 크기 제한
            if len(self.performance_trends) > 200:
                self.performance_trends = self.performance_trends[-100:]
                
        except Exception as e:
            print(f"통계 기록 오류: {e}")
    
    def record_template_usage(self, template_type: str, usage_count: int = 1, success_rate: float = None):
        """템플릿 사용 통계"""
        try:
            if template_type not in self.template_usage_stats:
                self.template_usage_stats[template_type] = {
                    "count": 0, 
                    "success_rate": 0.0, 
                    "last_used": None,
                    "effectiveness_score": 0.0
                }
            
            template_stat = self.template_usage_stats[template_type]
            template_stat["count"] += usage_count
            template_stat["last_used"] = datetime.now().isoformat()
            
            if success_rate is not None:
                # 가중 평균으로 성공률 업데이트
                current_weight = template_stat["count"] - usage_count
                if current_weight > 0:
                    template_stat["success_rate"] = (
                        (template_stat["success_rate"] * current_weight + success_rate * usage_count) 
                        / template_stat["count"]
                    )
                else:
                    template_stat["success_rate"] = success_rate
                
                # 효과성 점수 (사용 빈도와 성공률 결합)
                frequency_score = min(template_stat["count"] / 10, 1.0)  # 10회 사용을 100%로
                template_stat["effectiveness_score"] = (
                    template_stat["success_rate"] * 0.8 + frequency_score * 20
                )
                
        except Exception as e:
            print(f"템플릿 사용 통계 기록 오류: {e}")
    
    def record_memory_snapshot(self):
        """메모리 상태 기록"""
        if not self.memory_monitor_enabled:
            return
            
        try:
            memory_info = psutil.virtual_memory()
            process_info = psutil.Process()
            
            snapshot = {
                "timestamp": time.time(),
                "system_used_percent": memory_info.percent,
                "system_available_mb": memory_info.available / 1024 / 1024,
                "process_memory_mb": process_info.memory_info().rss / 1024 / 1024,
                "process_cpu_percent": process_info.cpu_percent(),
                "memory_efficiency": self._calculate_memory_efficiency(memory_info.percent)
            }
            
            self.memory_snapshots.append(snapshot)
            
            # 메모리 스냅샷 크기 제한
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots = self.memory_snapshots[-50:]
            
            # 메모리 사용량이 임계치 이상이면 경고 기록
            threshold = MEMORY_CONFIG.get("memory_threshold", 85)
            if memory_info.percent > threshold:
                self._log_memory_warning(snapshot)
                
        except Exception as e:
            print(f"메모리 스냅샷 기록 오류: {e}")
    
    def _calculate_memory_efficiency(self, memory_percent: float) -> float:
        """메모리 효율성 계산"""
        try:
            if memory_percent < 50:
                return 100.0
            elif memory_percent < 70:
                return 100 - (memory_percent - 50) * 2
            elif memory_percent < 85:
                return 60 - (memory_percent - 70) * 2
            else:
                return max(0, 30 - (memory_percent - 85) * 3)
        except Exception:
            return 50.0
    
    def get_domain_performance_summary(self) -> Dict:
        """도메인 성능 요약"""
        try:
            summary = {}
            
            for domain, stats in self.domain_stats.items():
                if stats["count"] == 0:
                    continue
                    
                success_rate = (stats["success_count"] / stats["count"]) * 100
                domain_weight = DOMAIN_WEIGHTS.get(domain, {"priority_boost": 1.0})
                weighted_score = success_rate * domain_weight["priority_boost"]
                
                summary[domain] = {
                    "success_rate": round(success_rate, 1),
                    "weighted_score": round(weighted_score, 1),
                    "efficiency_score": round(stats["efficiency_score"], 1),
                    "avg_time": round(stats["avg_time"], 2),
                    "question_count": stats["count"],
                    "trend": self.accuracy_tracking.get(domain, {}).get("trend", "unknown"),
                    "peak_accuracy": round(self.accuracy_tracking.get(domain, {}).get("peak_accuracy", 0), 1),
                    "current_streak": self.accuracy_tracking.get(domain, {}).get("current_streak", 0)
                }
            
            return summary
        except Exception as e:
            print(f"도메인 성능 요약 생성 오류: {e}")
            return {}
    
    def get_method_effectiveness_ranking(self) -> List[Dict]:
        """처리 방법 효과성 순위"""
        try:
            method_scores = []
            
            for method, stats in self.method_stats.items():
                if stats["count"] == 0:
                    continue
                    
                success_rate = (stats["success_count"] / stats["count"]) * 100
                time_score = max(0, 100 - (stats["avg_time"] * 3))  # 시간이 짧을수록 높은 점수
                usage_score = min(stats["count"] / 10 * 100, 100)  # 사용 빈도 점수
                
                overall_score = (
                    success_rate * 0.5 +       # 성공률 50%
                    time_score * 0.3 +         # 시간 효율 30%
                    usage_score * 0.2          # 사용 빈도 20%
                )
                
                method_scores.append({
                    "method": method,
                    "success_rate": round(success_rate, 1),
                    "avg_time": round(stats["avg_time"], 2),
                    "usage_count": stats["count"],
                    "overall_score": round(overall_score, 1),
                    "reliability_score": round(stats["reliability_score"], 1)
                })
            
            # 종합 점수 순으로 정렬
            method_scores.sort(key=lambda x: x["overall_score"], reverse=True)
            return method_scores
        except Exception as e:
            print(f"처리 방법 효과성 순위 생성 오류: {e}")
            return []
    
    def detect_performance_issues(self) -> List[Dict]:
        """성능 문제 탐지"""
        issues = []
        
        try:
            # 도메인별 성능 문제
            for domain, stats in self.domain_stats.items():
                if stats["count"] < 5:  # 충분한 데이터가 없으면 스킵
                    continue
                    
                success_rate = (stats["success_count"] / stats["count"]) * 100
                
                if success_rate < 50:
                    issues.append({
                        "type": "low_accuracy",
                        "domain": domain,
                        "current_rate": success_rate,
                        "severity": "high" if success_rate < 30 else "medium",
                        "suggestion": f"{domain} 도메인 특화 최적화 필요"
                    })
                
                if stats["avg_time"] > 25:
                    issues.append({
                        "type": "slow_processing",
                        "domain": domain,
                        "avg_time": stats["avg_time"],
                        "severity": "medium" if stats["avg_time"] < 35 else "high",
                        "suggestion": f"{domain} 도메인 처리 속도 최적화 필요"
                    })
                
                # 트렌드 문제
                acc_track = self.accuracy_tracking.get(domain, {})
                if acc_track.get("trend") == "declining":
                    issues.append({
                        "type": "declining_trend",
                        "domain": domain,
                        "trend": "declining",
                        "severity": "medium",
                        "suggestion": f"{domain} 도메인 성능 하락 추세 확인 필요"
                    })
            
            # 메모리 문제
            if self.memory_snapshots:
                recent_memory = self.memory_snapshots[-5:]  # 최근 5개
                avg_memory_usage = sum(snap["system_used_percent"] for snap in recent_memory) / len(recent_memory)
                
                if avg_memory_usage > 85:
                    issues.append({
                        "type": "high_memory_usage",
                        "current_usage": round(avg_memory_usage, 1),
                        "severity": "high" if avg_memory_usage > 90 else "medium",
                        "suggestion": "메모리 사용량 최적화 필요"
                    })
            
            # 오류 패턴 문제
            total_errors = sum(stat["count"] for stat in self.error_stats.values())
            total_questions = len(self.processing_times)
            
            if total_questions > 10 and total_errors / total_questions > 0.2:
                error_rate = (total_errors / total_questions) * 100
                issues.append({
                    "type": "high_error_rate",
                    "error_rate": round(error_rate, 1),
                    "severity": "high" if error_rate > 30 else "medium",
                    "suggestion": "오류 발생률이 높습니다. 오류 패턴 분석 필요"
                })
            
        except Exception as e:
            print(f"성능 문제 탐지 오류: {e}")
        
        return issues
    
    def generate_final_statistics(self, learning_data: Dict) -> Dict:
        """최종 통계 생성"""
        try:
            total_time = time.time() - self.start_time if self.start_time else 0
            total_questions = len(self.processing_times)
            
            # 기본 통계
            stats = {
                "execution_summary": {
                    "total_time_seconds": round(total_time, 2),
                    "total_time_minutes": round(total_time / 60, 2),
                    "total_questions": total_questions,
                    "avg_processing_time": round(sum(self.processing_times) / len(self.processing_times), 3) if self.processing_times else 0,
                    "questions_per_minute": round((total_questions / (total_time / 60)), 2) if total_time > 0 else 0,
                    "session_started": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None
                },
                "domain_performance": self.get_domain_performance_summary(),
                "method_effectiveness": self.get_method_effectiveness_ranking(),
                "question_type_analysis": self._analyze_question_types(),
                "performance_metrics": self._calculate_performance_metrics(),
                "learning_metrics": self._analyze_learning_data(learning_data),
                "system_metrics": self._analyze_system_metrics(),
                "error_analysis": self._analyze_errors(),
                "template_analysis": self._analyze_template_effectiveness(),
                "quality_metrics": self._calculate_quality_metrics(),
                "performance_issues": self.detect_performance_issues(),
                "recommendations": self._generate_recommendations(),
                "accuracy_trends": self.accuracy_tracking,
                "overall_score": self._calculate_overall_score()
            }
            
            self._write_detailed_log(stats)
            return stats
            
        except Exception as e:
            print(f"최종 통계 생성 오류: {e}")
            return self._generate_fallback_stats()
    
    def _analyze_question_types(self) -> Dict:
        """질문 타입 분석"""
        try:
            type_analysis = {}
            
            for q_type, stats in self.question_type_stats.items():
                if stats["count"] == 0:
                    continue
                    
                success_rate = (stats["success_count"] / stats["count"]) * 100
                
                type_analysis[q_type] = {
                    "question_count": stats["count"],
                    "success_rate": round(success_rate, 1),
                    "avg_processing_time": round(stats["avg_time"], 3),
                    "total_processing_time": round(stats["total_time"], 3),
                    "efficiency_rating": "high" if success_rate > 80 else "medium" if success_rate > 60 else "low"
                }
            
            return type_analysis
        except Exception as e:
            print(f"질문 타입 분석 오류: {e}")
            return {}
    
    def _analyze_template_effectiveness(self) -> Dict:
        """템플릿 효과성 분석"""
        try:
            if not self.template_usage_stats:
                return {"status": "no_template_data"}
            
            template_analysis = {}
            total_usage = sum(stat["count"] for stat in self.template_usage_stats.values())
            
            for template_type, stat in self.template_usage_stats.items():
                usage_ratio = (stat["count"] / total_usage) * 100 if total_usage > 0 else 0
                
                template_analysis[template_type] = {
                    "usage_count": stat["count"],
                    "usage_ratio": round(usage_ratio, 1),
                    "success_rate": round(stat["success_rate"], 1),
                    "effectiveness_score": round(stat["effectiveness_score"], 1),
                    "last_used": stat["last_used"],
                    "recommendation": self._get_template_recommendation(stat)
                }
            
            return template_analysis
        except Exception as e:
            print(f"템플릿 효과성 분석 오류: {e}")
            return {"status": "analysis_error"}
    
    def _get_template_recommendation(self, template_stat: Dict) -> str:
        """템플릿 추천사항"""
        try:
            effectiveness = template_stat["effectiveness_score"]
            usage_count = template_stat["count"]
            success_rate = template_stat["success_rate"]
            
            if effectiveness > 80:
                return "효과적인 템플릿입니다. 지속 사용 권장"
            elif effectiveness > 60:
                if usage_count < 5:
                    return "추가 활용을 통한 검증 필요"
                else:
                    return "적절한 효과성을 보입니다"
            elif effectiveness > 40:
                if success_rate < 60:
                    return "성공률 개선 필요"
                else:
                    return "사용 빈도 증가 고려"
            else:
                return "템플릿 개선 또는 대체 검토 필요"
        except Exception:
            return "분석 데이터 부족"
    
    def _generate_recommendations(self) -> List[Dict]:
        """성능 권장사항 생성"""
        recommendations = []
        
        try:
            # 도메인별 권장사항
            domain_summary = self.get_domain_performance_summary()
            
            for domain, perf in domain_summary.items():
                if perf["success_rate"] < 70:
                    recommendations.append({
                        "category": "domain_optimization",
                        "priority": "high" if perf["success_rate"] < 50 else "medium",
                        "domain": domain,
                        "current_rate": perf["success_rate"],
                        "recommendation": f"{domain} 도메인 정확도 향상 필요 (현재 {perf['success_rate']}%)",
                        "action": "도메인 특화 프롬프트 최적화 및 Few-shot 예시 보강"
                    })
                
                if perf["avg_time"] > 20:
                    recommendations.append({
                        "category": "performance_optimization",
                        "priority": "medium",
                        "domain": domain,
                        "current_time": perf["avg_time"],
                        "recommendation": f"{domain} 도메인 처리 속도 개선 필요 (현재 {perf['avg_time']}초)",
                        "action": "생성 파라미터 조정 및 캐시 활용 최적화"
                    })
            
            # 전체 성능 권장사항
            total_questions = len(self.processing_times)
            if total_questions > 0:
                total_success = sum(stat["success_count"] for stat in self.domain_stats.values())
                overall_rate = (total_success / total_questions) * 100
                
                if overall_rate < 70:
                    recommendations.append({
                        "category": "overall_improvement",
                        "priority": "high",
                        "current_rate": round(overall_rate, 1),
                        "target_rate": 70,
                        "recommendation": f"전체 정확도 개선 필요 (현재 {overall_rate:.1f}%, 목표 70%)",
                        "action": "패턴 분석 강화, 학습 데이터 품질 개선, 답변 검증 로직 보완"
                    })
            
            # 메서드 효과성 기반 권장사항
            method_ranking = self.get_method_effectiveness_ranking()
            if method_ranking:
                best_method = method_ranking[0]
                worst_methods = [m for m in method_ranking if m["overall_score"] < 60]
                
                if worst_methods:
                    recommendations.append({
                        "category": "method_optimization",
                        "priority": "medium",
                        "best_method": best_method["method"],
                        "worst_methods": [m["method"] for m in worst_methods],
                        "recommendation": f"효과적인 {best_method['method']} 방법의 활용도를 높이고, 저효율 방법들의 개선 필요",
                        "action": "방법별 가중치 조정 및 조건부 방법 선택 로직 개선"
                    })
            
            # 메모리 최적화 권장사항
            if self.memory_snapshots:
                avg_memory = sum(s["system_used_percent"] for s in self.memory_snapshots[-10:]) / min(10, len(self.memory_snapshots))
                if avg_memory > 80:
                    recommendations.append({
                        "category": "memory_optimization",
                        "priority": "medium" if avg_memory < 90 else "high",
                        "current_usage": round(avg_memory, 1),
                        "recommendation": f"메모리 사용량 최적화 필요 (현재 평균 {avg_memory:.1f}%)",
                        "action": "가비지 컬렉션 빈도 조정, 캐시 크기 제한, 메모리 누수 점검"
                    })
            
        except Exception as e:
            print(f"권장사항 생성 오류: {e}")
        
        return recommendations
    
    def _calculate_overall_score(self) -> Dict:
        """전체 성능 점수 계산"""
        try:
            if not self.domain_stats or len(self.processing_times) == 0:
                return {"score": 0, "grade": "F", "status": "insufficient_data"}
            
            # 정확도 점수 (40%)
            total_questions = len(self.processing_times)
            total_success = sum(stat["success_count"] for stat in self.domain_stats.values())
            accuracy_score = (total_success / total_questions) * 100 if total_questions > 0 else 0
            
            # 속도 점수 (25%)
            avg_time = sum(self.processing_times) / len(self.processing_times)
            speed_score = max(0, 100 - (avg_time * 4))  # 25초를 0점 기준으로
            
            # 일관성 점수 (20%)
            consistency_score = self._calculate_consistency_score()
            
            # 효율성 점수 (15%)
            efficiency_score = self._calculate_efficiency_score()
            
            # 가중 평균
            overall_score = (
                accuracy_score * 0.4 +
                speed_score * 0.25 +
                consistency_score * 0.2 +
                efficiency_score * 0.15
            )
            
            # 등급 산정
            if overall_score >= 90:
                grade = "A+"
            elif overall_score >= 85:
                grade = "A"
            elif overall_score >= 80:
                grade = "B+"
            elif overall_score >= 75:
                grade = "B"
            elif overall_score >= 70:
                grade = "C+"
            elif overall_score >= 60:
                grade = "C"
            elif overall_score >= 50:
                grade = "D"
            else:
                grade = "F"
            
            return {
                "score": round(overall_score, 1),
                "grade": grade,
                "components": {
                    "accuracy": round(accuracy_score, 1),
                    "speed": round(speed_score, 1),
                    "consistency": round(consistency_score, 1),
                    "efficiency": round(efficiency_score, 1)
                },
                "target_score": 70,
                "meets_target": overall_score >= 70
            }
        except Exception as e:
            print(f"전체 점수 계산 오류: {e}")
            return {"score": 0, "grade": "F", "status": "calculation_error"}
    
    def _clear_log_and_write_system_info(self):
        """로그 초기화 및 시스템 정보 기록"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] === 새로운 세션 시작 ===\n")
                
                if self.memory_monitor_enabled:
                    f.write(f"[{timestamp}] 시스템 메모리: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB\n")
                    f.write(f"[{timestamp}] CPU 코어: {psutil.cpu_count()}개\n")
                    f.write(f"[{timestamp}] 메모리 모니터링: 활성화\n")
                else:
                    f.write(f"[{timestamp}] 메모리 모니터링: 비활성화 (psutil 없음)\n")
                    
                f.write(f"[{timestamp}] Python 버전: {sys.version.split()[0]}\n")
                f.flush()
                
        except Exception as e:
            print(f"로그 초기화 및 시스템 정보 기록 오류: {e}")
    
    def _log_error_details(self, error_type: str, domain: str, method: str, processing_time: float):
        """오류 상세 정보 로그 기록"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] === 오류 발생 ===\n")
                f.write(f"[{timestamp}] 오류 유형: {error_type}\n")
                f.write(f"[{timestamp}] 도메인: {domain}\n")
                f.write(f"[{timestamp}] 처리 방법: {method}\n")
                f.write(f"[{timestamp}] 처리 시간: {processing_time:.3f}초\n")
                f.flush()
        except Exception as e:
            print(f"오류 상세 로그 기록 실패: {e}")
    
    def _log_memory_warning(self, snapshot: Dict):
        """메모리 경고 로그 기록"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] === 메모리 경고 ===\n")
                f.write(f"[{timestamp}] 시스템 메모리 사용률: {snapshot['system_used_percent']:.1f}%\n")
                f.write(f"[{timestamp}] 프로세스 메모리 사용량: {snapshot['process_memory_mb']:.1f}MB\n")
                f.write(f"[{timestamp}] CPU 사용률: {snapshot['process_cpu_percent']:.1f}%\n")
                f.write(f"[{timestamp}] 메모리 효율성: {snapshot['memory_efficiency']:.1f}%\n")
                f.flush()
        except Exception as e:
            print(f"메모리 경고 로그 기록 실패: {e}")
    
    def _analyze_errors(self) -> Dict:
        """오류 분석"""
        if not self.error_stats:
            return {"status": "no_errors"}
            
        try:
            error_analysis = {}
            total_errors = sum(error["count"] for error in self.error_stats.values())
            
            for error_type, stats in self.error_stats.items():
                error_analysis[error_type] = {
                    "count": stats["count"],
                    "percentage": round((stats["count"] / total_errors) * 100, 1) if total_errors > 0 else 0,
                    "avg_processing_time": round(stats["avg_time"], 3),
                    "total_processing_time": round(stats["total_time"], 3),
                    "domain_distribution": stats["domain_distribution"],
                    "severity": "high" if stats["count"] > total_errors * 0.3 else "medium" if stats["count"] > total_errors * 0.1 else "low"
                }
            
            error_analysis["summary"] = {
                "total_errors": total_errors,
                "error_rate": round((total_errors / len(self.processing_times)) * 100, 1) if self.processing_times else 0,
                "most_common_error": max(self.error_stats.items(), key=lambda x: x[1]["count"])[0] if self.error_stats else None
            }
            
            return error_analysis
        except Exception as e:
            print(f"오류 분석 실패: {e}")
            return {"status": "analysis_error"}
    
    def _calculate_performance_metrics(self) -> Dict:
        """성능 지표 계산"""
        if not self.processing_times:
            return {"status": "no_data"}
            
        try:
            processing_times = sorted(self.processing_times)
            
            return {
                "min_processing_time": round(min(processing_times), 3),
                "max_processing_time": round(max(processing_times), 3),
                "median_processing_time": round(processing_times[len(processing_times) // 2], 3),
                "p95_processing_time": round(processing_times[int(len(processing_times) * 0.95)], 3),
                "p99_processing_time": round(processing_times[int(len(processing_times) * 0.99)], 3),
                "std_deviation": round(self._calculate_std_deviation(processing_times), 3),
                "efficiency_score": self._calculate_efficiency_score(),
                "throughput_score": self._calculate_throughput_score(),
                "consistency_score": self._calculate_consistency_score(),
                "performance_stability": self._assess_performance_stability()
            }
        except Exception as e:
            print(f"성능 지표 계산 오류: {e}")
            return {"status": "calculation_error"}
    
    def _assess_performance_stability(self) -> str:
        """성능 안정성 평가"""
        try:
            if len(self.processing_times) < 10:
                return "데이터 부족"
            
            recent_times = self.processing_times[-10:]
            earlier_times = self.processing_times[-20:-10] if len(self.processing_times) >= 20 else self.processing_times[:-10]
            
            if not earlier_times:
                return "평가 불가"
            
            recent_avg = sum(recent_times) / len(recent_times)
            earlier_avg = sum(earlier_times) / len(earlier_times)
            
            change_rate = abs(recent_avg - earlier_avg) / earlier_avg
            
            if change_rate < 0.1:
                return "안정적"
            elif change_rate < 0.25:
                return "보통"
            else:
                return "불안정"
        except Exception:
            return "평가 오류"
    
    def _calculate_quality_metrics(self) -> Dict:
        """품질 지표 계산"""
        try:
            total_questions = len(self.processing_times)
            total_success = sum(stats["success_count"] for stats in self.domain_stats.values())
            total_errors = sum(stats["count"] for stats in self.error_stats.values())
            
            overall_success_rate = (total_success / total_questions) * 100 if total_questions > 0 else 0
            error_rate = (total_errors / total_questions) * 100 if total_questions > 0 else 0
            
            # 품질 안정성 평가
            domain_rates = []
            for stats in self.domain_stats.values():
                if stats["count"] > 0:
                    rate = (stats["success_count"] / stats["count"]) * 100
                    domain_rates.append(rate)
            
            quality_consistency = 100 - (self._calculate_std_deviation(domain_rates) if len(domain_rates) > 1 else 0)
            
            return {
                "overall_success_rate": round(overall_success_rate, 1),
                "error_rate": round(error_rate, 1),
                "template_utilization": len(self.template_usage_stats),
                "domain_coverage": len(self.domain_stats),
                "method_diversity": len(self.method_stats),
                "quality_consistency": round(max(0, quality_consistency), 1),
                "meets_target": overall_success_rate >= 70,
                "quality_grade": "excellent" if overall_success_rate >= 85 else 
                              "good" if overall_success_rate >= 75 else 
                              "acceptable" if overall_success_rate >= 65 else 
                              "needs_improvement"
            }
        except Exception as e:
            print(f"품질 지표 계산 오류: {e}")
            return {"status": "calculation_error"}
    
    def _analyze_learning_data(self, learning_data: Dict) -> Dict:
        """학습 데이터 분석"""
        try:
            total_successful = learning_data.get("successful_answers", 0)
            total_failed = learning_data.get("failed_answers", 0)
            total_patterns = learning_data.get("question_patterns", 0)
            
            learning_rate = (total_successful / (total_successful + total_failed)) * 100 if (total_successful + total_failed) > 0 else 0
            
            return {
                "successful_answers": total_successful,
                "failed_answers": total_failed,
                "pattern_records": total_patterns,
                "learning_rate": round(learning_rate, 1),
                "knowledge_accumulation": total_successful + total_patterns,
                "error_recovery_data": total_failed,
                "learning_efficiency": round((total_successful / len(self.processing_times)) * 100, 1) if self.processing_times else 0,
                "data_quality": "high" if learning_rate > 80 else "medium" if learning_rate > 60 else "low",
                "improvement_potential": round(100 - learning_rate, 1) if learning_rate < 100 else 0
            }
        except Exception as e:
            print(f"학습 데이터 분석 오류: {e}")
            return {"status": "analysis_error"}
    
    def _analyze_system_metrics(self) -> Dict:
        """시스템 지표 분석"""
        if not self.memory_snapshots or not self.memory_monitor_enabled:
            return {"monitoring_status": "disabled" if not self.memory_monitor_enabled else "no_data"}
            
        try:
            system_memory_usages = [snap["system_used_percent"] for snap in self.memory_snapshots]
            process_memory_usages = [snap["process_memory_mb"] for snap in self.memory_snapshots]
            cpu_usages = [snap.get("process_cpu_percent", 0) for snap in self.memory_snapshots]
            memory_efficiencies = [snap.get("memory_efficiency", 50) for snap in self.memory_snapshots]
            
            return {
                "monitoring_status": "active",
                "avg_system_memory_usage": round(sum(system_memory_usages) / len(system_memory_usages), 1),
                "peak_system_memory_usage": round(max(system_memory_usages), 1),
                "avg_process_memory_usage": round(sum(process_memory_usages) / len(process_memory_usages), 1),
                "peak_process_memory_usage": round(max(process_memory_usages), 1),
                "avg_cpu_usage": round(sum(cpu_usages) / len(cpu_usages), 1),
                "peak_cpu_usage": round(max(cpu_usages), 1),
                "avg_memory_efficiency": round(sum(memory_efficiencies) / len(memory_efficiencies), 1),
                "memory_snapshots": len(self.memory_snapshots),
                "system_stability": "정상" if max(system_memory_usages) < 85 else "주의필요"
            }
        except Exception as e:
            print(f"시스템 지표 분석 오류: {e}")
            return {"monitoring_status": "error"}
    
    def _calculate_std_deviation(self, values: List[float]) -> float:
        """표준편차 계산"""
        if len(values) < 2:
            return 0
            
        try:
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return variance ** 0.5
        except Exception:
            return 0
    
    def _calculate_efficiency_score(self) -> float:
        """처리 효율 점수 계산"""
        if not self.processing_times:
            return 0
            
        try:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            consistency = 1 - (self._calculate_std_deviation(self.processing_times) / avg_time) if avg_time > 0 else 0
            speed_score = max(0, 1 - (avg_time / 15))  # 15초를 기준으로 속도 점수
            
            return round((consistency * 0.6 + speed_score * 0.4) * 100, 1)
        except Exception:
            return 0
    
    def _calculate_throughput_score(self) -> float:
        """처리량 점수 계산"""
        try:
            total_time = time.time() - self.start_time if self.start_time else 0
            total_questions = len(self.processing_times)
            
            if total_time > 0:
                questions_per_minute = (total_questions / (total_time / 60))
                # 분당 25문항을 100점으로 기준
                return round(min((questions_per_minute / 25) * 100, 100), 1)
            return 0
        except Exception:
            return 0
    
    def _calculate_consistency_score(self) -> float:
        """일관성 점수 계산"""
        if not self.processing_times or len(self.processing_times) < 2:
            return 0
            
        try:
            std_dev = self._calculate_std_deviation(self.processing_times)
            avg_time = sum(self.processing_times) / len(self.processing_times)
            
            if avg_time > 0:
                consistency_ratio = 1 - (std_dev / avg_time)
                return round(max(0, consistency_ratio) * 100, 1)
            return 0
        except Exception:
            return 0
    
    def _generate_fallback_stats(self) -> Dict:
        """대체 통계 생성"""
        return {
            "execution_summary": {
                "total_time_seconds": 0,
                "total_time_minutes": 0,
                "total_questions": 0,
                "avg_processing_time": 0,
                "questions_per_minute": 0
            },
            "error": "통계 생성 중 오류 발생"
        }
    
    def _write_detailed_log(self, stats: Dict):
        """상세 로그 작성"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 실행 요약
                exec_summary = stats.get("execution_summary", {})
                f.write(f"[{timestamp}] === 실행 통계 ===\n")
                f.write(f"[{timestamp}] 총 처리 시간: {exec_summary.get('total_time_minutes', 0)}분 ({exec_summary.get('total_time_seconds', 0)}초)\n")
                f.write(f"[{timestamp}] 처리 문항 수: {exec_summary.get('total_questions', 0)}개\n")
                f.write(f"[{timestamp}] 평균 문항 처리 시간: {exec_summary.get('avg_processing_time', 0)}초\n")
                f.write(f"[{timestamp}] 분당 처리량: {exec_summary.get('questions_per_minute', 0)}문항/분\n")
                
                # 전체 성능 점수
                overall_score = stats.get("overall_score", {})
                if overall_score and overall_score.get("score", 0) > 0:
                    f.write(f"[{timestamp}] === 전체 성능 평가 ===\n")
                    f.write(f"[{timestamp}] 전체 점수: {overall_score.get('score', 0)}점 ({overall_score.get('grade', 'F')})\n")
                    f.write(f"[{timestamp}] 목표 달성: {'Yes' if overall_score.get('meets_target', False) else 'No'}\n")
                    components = overall_score.get("components", {})
                    f.write(f"[{timestamp}] 정확도: {components.get('accuracy', 0)}점, 속도: {components.get('speed', 0)}점\n")
                    f.write(f"[{timestamp}] 일관성: {components.get('consistency', 0)}점, 효율성: {components.get('efficiency', 0)}점\n")
                
                # 도메인별 성능
                domain_perf = stats.get("domain_performance", {})
                if domain_perf:
                    f.write(f"[{timestamp}] === 도메인별 성능 ===\n")
                    for domain, perf in domain_perf.items():
                        f.write(f"[{timestamp}] {domain}: {perf.get('success_rate', 0)}% "
                               f"(효율성 {perf.get('efficiency_score', 0)}점, 평균 {perf.get('avg_time', 0)}초, "
                               f"트렌드 {perf.get('trend', 'unknown')}, 연속성공 {perf.get('current_streak', 0)}회)\n")
                
                # 처리 방법 효과성
                method_eff = stats.get("method_effectiveness", [])
                if method_eff:
                    f.write(f"[{timestamp}] === 처리 방법 효과성 ===\n")
                    for method in method_eff[:3]:  # 상위 3개만
                        f.write(f"[{timestamp}] {method.get('method', 'unknown')}: "
                               f"{method.get('success_rate', 0)}% 성공률, "
                               f"{method.get('overall_score', 0)}점 종합점수\n")
                
                # 성능 문제
                issues = stats.get("performance_issues", [])
                if issues:
                    f.write(f"[{timestamp}] === 발견된 성능 문제 ===\n")
                    for issue in issues:
                        f.write(f"[{timestamp}] {issue.get('type', 'unknown')} ({issue.get('severity', 'unknown')} 심각도): "
                               f"{issue.get('suggestion', 'No suggestion')}\n")
                
                # 권장사항
                recommendations = stats.get("recommendations", [])
                if recommendations:
                    f.write(f"[{timestamp}] === 개선 권장사항 ===\n")
                    high_priority = [r for r in recommendations if r.get("priority") == "high"]
                    for rec in high_priority[:3]:  # 고우선순위 3개만
                        f.write(f"[{timestamp}] [높음] {rec.get('recommendation', 'No recommendation')}\n")
                        f.write(f"[{timestamp}]        액션: {rec.get('action', 'No action')}\n")
                
                # 품질 지표
                quality = stats.get("quality_metrics", {})
                if quality and quality.get("overall_success_rate", 0) > 0:
                    f.write(f"[{timestamp}] === 품질 지표 ===\n")
                    f.write(f"[{timestamp}] 전체 성공률: {quality.get('overall_success_rate', 0)}%\n")
                    f.write(f"[{timestamp}] 오류율: {quality.get('error_rate', 0)}%\n")
                    f.write(f"[{timestamp}] 품질 등급: {quality.get('quality_grade', 'unknown')}\n")
                    f.write(f"[{timestamp}] 목표 달성: {'Yes' if quality.get('meets_target', False) else 'No'}\n")
                
                # 학습 지표
                learning = stats.get("learning_metrics", {})
                if learning and learning.get("successful_answers", 0) > 0:
                    f.write(f"[{timestamp}] === 학습 데이터 현황 ===\n")
                    f.write(f"[{timestamp}] 성공 답변: {learning.get('successful_answers', 0)}개\n")
                    f.write(f"[{timestamp}] 실패 답변: {learning.get('failed_answers', 0)}개\n")
                    f.write(f"[{timestamp}] 패턴 기록: {learning.get('pattern_records', 0)}개\n")
                    f.write(f"[{timestamp}] 학습 성공률: {learning.get('learning_rate', 0)}%\n")
                    f.write(f"[{timestamp}] 데이터 품질: {learning.get('data_quality', 'unknown')}\n")
                
                # 시스템 지표
                system = stats.get("system_metrics", {})
                if system and system.get("monitoring_status") == "active":
                    f.write(f"[{timestamp}] === 시스템 성능 ===\n")
                    f.write(f"[{timestamp}] 평균 시스템 메모리: {system.get('avg_system_memory_usage', 0)}%\n")
                    f.write(f"[{timestamp}] 최대 시스템 메모리: {system.get('peak_system_memory_usage', 0)}%\n")
                    f.write(f"[{timestamp}] 평균 프로세스 메모리: {system.get('avg_process_memory_usage', 0)}MB\n")
                    f.write(f"[{timestamp}] 평균 메모리 효율성: {system.get('avg_memory_efficiency', 0)}%\n")
                    f.write(f"[{timestamp}] 시스템 상태: {system.get('system_stability', '알 수 없음')}\n")
                
                f.write(f"[{timestamp}] === 세션 종료 ===\n\n")
                f.flush()
                
        except Exception as e:
            print(f"상세 로그 작성 오류: {e}")
            # 기본 로그라도 작성 시도
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] 상세 로그 작성 실패: {e}\n")
                    f.write(f"[{timestamp}] === 세션 종료 (오류) ===\n\n")
            except Exception:
                pass