"""
Simple Event Bus for Component Communication

Enables decoupled communication between SCFD backbone components:
- Parameter registry parameter updates
- Sensing router signal changes  
- EFE tuner optimization results
- Vector adapter type detection

Thread-safe with automatic cleanup and type-safe event definitions.
"""

import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from collections import defaultdict
import uuid
import time

class EventType(Enum):
    """Core event types for SCFD system communication"""
    PARAMETER_UPDATED = auto()
    PARAMETER_CONSTRAINT_VIOLATED = auto()
    SIGNAL_THRESHOLD_CROSSED = auto()
    VECTOR_TYPE_DETECTED = auto()
    COMPOSITION_RESULT = auto()
    EFE_OPTIMIZATION_COMPLETE = auto()
    CONTROLLER_NUDGE_APPLIED = auto()
    SYSTEM_HEALTH_CHECK = auto()

@dataclass(frozen=True)
class Event:
    """Base event class with automatic timestamping and correlation"""
    event_type: EventType
    source_component: str
    data: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.correlation_id is None:
            object.__setattr__(self, 'correlation_id', str(uuid.uuid4())[:8])

# Specific event data classes for type safety
@dataclass(frozen=True) 
class ParameterUpdateEvent(Event):
    """Parameter registry update notification"""
    def __post_init__(self):
        super().__post_init__()
        required_fields = {'parameter_name', 'old_value', 'new_value', 'role'}
        if not required_fields.issubset(self.data.keys()):
            raise ValueError(f"Missing required fields: {required_fields - self.data.keys()}")

@dataclass(frozen=True)
class SignalThresholdEvent(Event):
    """Sensing router threshold crossing notification"""
    def __post_init__(self):
        super().__post_init__()
        required_fields = {'signal_name', 'threshold_value', 'current_value', 'direction'}
        if not required_fields.issubset(self.data.keys()):
            raise ValueError(f"Missing required fields: {required_fields - self.data.keys()}")

@dataclass(frozen=True)
class EFEOptimizationEvent(Event):
    """EFE tuner optimization completion notification"""
    def __post_init__(self):
        super().__post_init__()
        required_fields = {'optimization_strategy', 'parameters_updated', 'efe_improvement'}
        if not required_fields.issubset(self.data.keys()):
            raise ValueError(f"Missing required fields: {required_fields - self.data.keys()}")

# Event handler type definitions
EventHandler = Callable[[Event], None]
T = TypeVar('T', bound=Event)

class EventSubscription:
    """Manages event subscription with automatic cleanup"""
    
    def __init__(self, event_type: EventType, handler: EventHandler, 
                 subscriber_id: str, weak_ref: bool = True):
        self.event_type = event_type
        self.subscriber_id = subscriber_id
        self.subscription_id = str(uuid.uuid4())[:8]
        self.created_at = time.time()
        
        # Use weak references to prevent memory leaks
        if weak_ref and hasattr(handler, '__self__'):
            self._handler_ref = weakref.WeakMethod(handler)
        else:
            self._handler_ref = handler
            
    def get_handler(self) -> Optional[EventHandler]:
        """Get handler, returns None if weak reference is dead"""
        if isinstance(self._handler_ref, weakref.WeakMethod):
            return self._handler_ref()
        return self._handler_ref
    
    def is_alive(self) -> bool:
        """Check if subscription is still valid"""
        return self.get_handler() is not None

class EventBus:
    """
    Thread-safe event bus for SCFD component communication
    
    Features:
    - Type-safe event publishing/subscribing
    - Automatic cleanup of dead weak references
    - Correlation tracking for event chains
    - Performance monitoring with handler timing
    - Subscription lifecycle management
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._subscriptions: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        self._subscription_count = 0
        self._event_count = 0
        self._cleanup_threshold = 100  # Clean up dead refs every N events
        
    def subscribe(self, event_type: EventType, handler: EventHandler, 
                 subscriber_id: str, weak_ref: bool = True) -> str:
        """
        Subscribe to events of specific type
        
        Args:
            event_type: Type of events to receive
            handler: Callback function to handle events
            subscriber_id: Identifier for the subscribing component
            weak_ref: Use weak references (default True) to prevent memory leaks
            
        Returns:
            Subscription ID for later unsubscription
        """
        with self._lock:
            subscription = EventSubscription(event_type, handler, subscriber_id, weak_ref)
            self._subscriptions[event_type].append(subscription)
            self._subscription_count += 1
            return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe using subscription ID
        
        Returns:
            True if subscription was found and removed
        """
        with self._lock:
            for event_type, subscriptions in self._subscriptions.items():
                for i, sub in enumerate(subscriptions):
                    if sub.subscription_id == subscription_id:
                        del subscriptions[i]
                        return True
            return False
    
    def unsubscribe_component(self, subscriber_id: str) -> int:
        """
        Unsubscribe all subscriptions for a component
        
        Returns:
            Number of subscriptions removed
        """
        with self._lock:
            removed_count = 0
            for event_type, subscriptions in self._subscriptions.items():
                self._subscriptions[event_type] = [
                    sub for sub in subscriptions 
                    if sub.subscriber_id != subscriber_id
                ]
                removed_count += len(subscriptions) - len(self._subscriptions[event_type])
            return removed_count
    
    def publish(self, event: Event) -> int:
        """
        Publish event to all subscribers
        
        Returns:
            Number of handlers that received the event
        """
        with self._lock:
            self._event_count += 1
            
            # Periodic cleanup of dead weak references
            if self._event_count % self._cleanup_threshold == 0:
                self._cleanup_dead_subscriptions()
            
            handlers_notified = 0
            subscriptions = self._subscriptions.get(event.event_type, [])
            
            for subscription in subscriptions[:]:  # Copy to avoid modification during iteration
                handler = subscription.get_handler()
                if handler is None:
                    # Dead weak reference, remove it
                    subscriptions.remove(subscription)
                    continue
                
                try:
                    handler(event)
                    handlers_notified += 1
                except Exception as e:
                    # Log error but continue notifying other handlers
                    print(f"Event handler error in {subscription.subscriber_id}: {e}")
            
            return handlers_notified
    
    def publish_parameter_update(self, source: str, parameter_name: str, 
                               old_value: Any, new_value: Any, role: str,
                               correlation_id: Optional[str] = None) -> int:
        """Convenience method for parameter update events"""
        event = ParameterUpdateEvent(
            event_type=EventType.PARAMETER_UPDATED,
            source_component=source,
            data={
                'parameter_name': parameter_name,
                'old_value': old_value,
                'new_value': new_value,
                'role': role
            },
            timestamp=time.time(),
            correlation_id=correlation_id
        )
        return self.publish(event)
    
    def publish_signal_threshold(self, source: str, signal_name: str,
                               threshold_value: float, current_value: float,
                               direction: str, correlation_id: Optional[str] = None) -> int:
        """Convenience method for signal threshold events"""
        event = SignalThresholdEvent(
            event_type=EventType.SIGNAL_THRESHOLD_CROSSED,
            source_component=source,
            data={
                'signal_name': signal_name,
                'threshold_value': threshold_value,
                'current_value': current_value,
                'direction': direction  # 'above' or 'below'
            },
            timestamp=time.time(),
            correlation_id=correlation_id
        )
        return self.publish(event)
    
    def publish_efe_optimization(self, source: str, strategy: str,
                               parameters_updated: Dict[str, Any],
                               efe_improvement: float,
                               correlation_id: Optional[str] = None) -> int:
        """Convenience method for EFE optimization events"""
        event = EFEOptimizationEvent(
            event_type=EventType.EFE_OPTIMIZATION_COMPLETE,
            source_component=source,
            data={
                'optimization_strategy': strategy,
                'parameters_updated': parameters_updated,
                'efe_improvement': efe_improvement
            },
            timestamp=time.time(),
            correlation_id=correlation_id
        )
        return self.publish(event)
    
    def _cleanup_dead_subscriptions(self):
        """Remove subscriptions with dead weak references"""
        total_cleaned = 0
        for event_type, subscriptions in self._subscriptions.items():
            original_count = len(subscriptions)
            self._subscriptions[event_type] = [
                sub for sub in subscriptions if sub.is_alive()
            ]
            total_cleaned += original_count - len(self._subscriptions[event_type])
        
        if total_cleaned > 0:
            print(f"EventBus: Cleaned up {total_cleaned} dead subscriptions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        with self._lock:
            subscription_counts = {
                event_type.name: len(subs) 
                for event_type, subs in self._subscriptions.items()
            }
            
            return {
                'total_subscriptions': sum(len(subs) for subs in self._subscriptions.values()),
                'subscription_counts_by_event': subscription_counts,
                'total_events_published': self._event_count,
                'event_types_supported': len(EventType)
            }

# Global event bus singleton for SCFD system
_global_event_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()

def get_event_bus() -> EventBus:
    """Get the global SCFD event bus singleton"""
    global _global_event_bus
    if _global_event_bus is None:
        with _bus_lock:
            if _global_event_bus is None:
                _global_event_bus = EventBus()
    return _global_event_bus

def reset_event_bus():
    """Reset the global event bus (primarily for testing)"""
    global _global_event_bus
    with _bus_lock:
        _global_event_bus = None

# Decorator for automatic event publishing
def publish_on_change(event_type: EventType, source_component: str):
    """Decorator to automatically publish events when method completes"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            
            # Publish generic event with method args as data
            event_data = {
                'method_name': func.__name__,
                'args': args,
                'kwargs': kwargs,
                'result': result
            }
            
            event = Event(
                event_type=event_type,
                source_component=source_component,
                data=event_data,
                timestamp=time.time()
            )
            
            get_event_bus().publish(event)
            return result
        return wrapper
    return decorator

class EventBusComponent(ABC):
    """
    Base class for SCFD components that use the event bus
    
    Provides automatic subscription management and cleanup
    """
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self._subscription_ids: List[str] = []
        self._event_bus = get_event_bus()
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> str:
        """Subscribe to events with automatic cleanup on component destruction"""
        subscription_id = self._event_bus.subscribe(
            event_type, handler, self.component_id
        )
        self._subscription_ids.append(subscription_id)
        return subscription_id
    
    def publish(self, event: Event) -> int:
        """Publish event through the event bus"""
        return self._event_bus.publish(event)
    
    def cleanup_subscriptions(self):
        """Cleanup all subscriptions for this component"""
        for subscription_id in self._subscription_ids:
            self._event_bus.unsubscribe(subscription_id)
        self._subscription_ids.clear()
    
    def __del__(self):
        """Automatic cleanup on destruction"""
        try:
            self.cleanup_subscriptions()
        except:
            pass  # Ignore errors during cleanup

if __name__ == "__main__":
    # Simple test of event bus functionality
    bus = get_event_bus()
    
    def test_handler(event: Event):
        print(f"Received {event.event_type.name} from {event.source_component}: {event.data}")
    
    # Subscribe to parameter updates
    sub_id = bus.subscribe(EventType.PARAMETER_UPDATED, test_handler, "test_component")
    
    # Publish a test event
    bus.publish_parameter_update(
        source="test_source",
        parameter_name="alpha", 
        old_value=0.5,
        new_value=0.7,
        role="STABILITY"
    )
    
    # Check stats
    print("Event bus stats:", bus.get_stats())
    
    # Cleanup
    bus.unsubscribe(sub_id)