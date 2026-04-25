---
name: analytics-dashboard
description: User behavior tracking and analytics dashboards
---

# Analytics Dashboard Skill

Automate setup of analytics tracking and dashboard generation for web applications.

---

## Purpose

This skill helps you implement comprehensive analytics tracking and create beautiful dashboards to understand user behavior, track conversions, and measure product success.

---

## When to Use

- Adding analytics to a new application
- Implementing event tracking for user actions
- Creating analytics dashboards
- Switching analytics providers
- Setting up conversion tracking
- Building admin analytics views

---

## What It Does

1. **Detects project type** (Next.js, React, Express, Python)
2. **Installs analytics dependencies** (PostHog, Mixpanel, Amplitude, Segment)
3. **Creates analytics configuration** with initialization code
4. **Generates dashboard components** with key metrics
5. **Provides tracking patterns** for common events
6. **Includes best practices** for performance and privacy

---

## Quick Start

### 1. Run Setup Script

```bash
python3 /home/ubuntu/skills/analytics-dashboard/scripts/setup_analytics.py \
  /path/to/your/project \
  posthog \
  YOUR_API_KEY
```

**Arguments**:
- `project_dir`: Path to your project
- `provider`: Analytics provider (posthog, mixpanel, amplitude, segment)
- `api_key`: Your analytics API key

---

### 2. Initialize Analytics

**Next.js** - Add to `app/layout.tsx`:
```typescript
import { initAnalytics } from '@/lib/analytics'

export default function RootLayout({ children }) {
  useEffect(() => {
    initAnalytics()
  }, [])
  
  return <html>{children}</html>
}
```

**Python/Flask** - Add to app initialization:
```python
from analytics import track_event, identify_user

@app.before_request
def track_page_view():
    if current_user.is_authenticated:
        track_event(current_user.id, 'page_view', {
            'path': request.path
        })
```

---

### 3. Track Events

```typescript
import { trackEvent } from '@/lib/analytics'

// Track button click
<button onClick={() => trackEvent('signup_clicked', {
  location: 'homepage'
})}>
  Sign Up
</button>

// Track form submission
const handleSubmit = (data) => {
  trackEvent('form_submitted', {
    form_name: 'contact',
    fields: Object.keys(data)
  })
}

// Track conversion
trackEvent('purchase_completed', {
  amount: 99.99,
  currency: 'USD',
  items: 3
})
```

---

### 4. Add Dashboard

Copy the generated dashboard component:
```bash
cp /home/ubuntu/skills/analytics-dashboard/templates/dashboard.tsx \
   /path/to/your/project/components/
```

Use in your app:
```typescript
import AnalyticsDashboard from '@/components/AnalyticsDashboard'

export default function AdminPage() {
  return <AnalyticsDashboard />
}
```

---

## Supported Providers

### PostHog (Recommended)
- **Best for**: Startups, product analytics
- **Features**: Session replay, feature flags, A/B testing
- **Pricing**: Generous free tier
- **Setup time**: 5 minutes

### Mixpanel
- **Best for**: User behavior analysis
- **Features**: Funnels, retention, cohorts
- **Pricing**: Free up to 100k events/month
- **Setup time**: 10 minutes

### Amplitude
- **Best for**: Product-led growth
- **Features**: Behavioral analytics, predictions
- **Pricing**: Free up to 10M events/month
- **Setup time**: 10 minutes

### Segment
- **Best for**: Multiple analytics tools
- **Features**: Single API for all providers
- **Pricing**: Free up to 1k users/month
- **Setup time**: 15 minutes

---

## Key Metrics to Track

### Engagement
- Daily Active Users (DAU)
- Weekly Active Users (WAU)
- Monthly Active Users (MAU)
- Session duration
- Pages per session

### Conversion
- Signup conversion rate
- Trial-to-paid conversion
- Purchase completion rate
- Funnel drop-off points

### Retention
- Day 1, 7, 30 retention
- Churn rate
- Cohort analysis

### Product
- Feature adoption rate
- User flows
- A/B test results

---

## Event Tracking Patterns

### Page Views
```typescript
useEffect(() => {
  trackEvent('page_view', {
    path: window.location.pathname,
    referrer: document.referrer
  })
}, [pathname])
```

### User Actions
```typescript
trackEvent('button_clicked', {
  button_name: 'signup',
  location: 'homepage_hero'
})
```

### User Identification
```typescript
identifyUser(user.id, {
  email: user.email,
  plan: user.subscription_plan,
  signup_date: user.created_at
})
```

### Conversions
```typescript
trackEvent('purchase_completed', {
  amount: total,
  currency: 'USD',
  payment_method: 'stripe'
})
```

---

## Dashboard Components

The generated dashboard includes:

1. **Key Metrics Cards**
   - Total events
   - Unique users
   - Average session duration
   - Bounce rate

2. **Charts**
   - User growth (line chart)
   - Top events (bar chart)
   - Conversion funnel

3. **Tables**
   - Top pages by views
   - Top events by count
   - Recent user activity

---

## Best Practices

### Performance
- ✅ Load analytics asynchronously
- ✅ Batch events together
- ✅ Debounce high-frequency events
- ❌ Don't block page load

### Privacy
- ✅ Get user consent (GDPR)
- ✅ Provide opt-out mechanism
- ✅ Anonymize sensitive data
- ❌ Don't track PII without consent

### Data Quality
- ✅ Include context with events
- ✅ Use consistent naming
- ✅ Test in development
- ❌ Don't track too much

---

## Files Generated

### Scripts
- `scripts/setup_analytics.py` - Automated setup script

### Templates
- `templates/dashboard.tsx` - Analytics dashboard component

### References
- `references/analytics_patterns.md` - Comprehensive guide (300+ lines)

---

## Workflow

1. **Setup** → Run script to install and configure
2. **Initialize** → Add analytics to app entry point
3. **Track** → Add event tracking throughout app
4. **Dashboard** → Add dashboard component for viewing
5. **Optimize** → Refine based on data

---

## Examples

### E-commerce Site
```typescript
// Product viewed
trackEvent('product_viewed', {
  product_id: product.id,
  category: product.category,
  price: product.price
})

// Added to cart
trackEvent('add_to_cart', {
  product_id: product.id,
  quantity: 1,
  cart_total: cart.total
})

// Purchase completed
trackEvent('purchase_completed', {
  order_id: order.id,
  total: order.total,
  items: order.items.length
})
```

### SaaS Application
```typescript
// Trial started
trackEvent('trial_started', {
  plan: 'pro',
  trial_days: 14
})

// Feature used
trackEvent('feature_used', {
  feature_name: 'export_data',
  usage_count: user.export_count
})

// Subscription upgraded
trackEvent('subscription_upgraded', {
  from_plan: 'basic',
  to_plan: 'pro',
  mrr_change: 20
})
```

---

## Troubleshooting

### Events not showing up
- Check API key is correct
- Verify analytics is initialized
- Check browser console for errors
- Test in production mode

### Dashboard not loading
- Verify API endpoint exists
- Check CORS settings
- Ensure data is being tracked
- Check network tab for errors

### Performance issues
- Batch events together
- Debounce high-frequency events
- Load analytics asynchronously
- Use sampling for high-volume

---

## Time Savings

**Manual Setup**: 2-3 hours
- Research providers: 30 min
- Install dependencies: 15 min
- Write configuration: 45 min
- Create dashboard: 60 min
- Test and debug: 30 min

**With This Skill**: 15-20 minutes
- Run setup script: 5 min
- Configure API keys: 5 min
- Add tracking: 5 min
- Test: 5 min

**Time Saved**: ~2 hours per project

---

## Related Skills

- `deployment-automation` - Deploy with analytics configured
- `user-authentication-system` - Identify users after login
- `feature-flag-system` - Track feature flag usage
- `error-monitoring-setup` - Combine with error tracking

---

## Resources

- [PostHog Documentation](https://posthog.com/docs)
- [Mixpanel Best Practices](https://mixpanel.com/blog)
- [Amplitude Playbook](https://amplitude.com/blog)
- [Analytics Patterns Reference](./references/analytics_patterns.md)

---

*Built with best practices, privacy-conscious, performance-optimized.*
