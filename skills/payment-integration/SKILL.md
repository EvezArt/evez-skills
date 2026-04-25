---
name: payment-integration
description: Stripe payment integration with checkout and subscriptions
---

# payment-integration

Automate Stripe payment integration for one-time payments and subscriptions.

## What It Does

- Sets up Stripe SDK and configuration
- Generates checkout components
- Creates webhook handlers
- Provides subscription management
- **Time saved**: ~2 hours per implementation

## When to Use

- Adding payments to any application
- Building SaaS with subscriptions
- Selling digital products
- Implementing paywalls

## Quick Start

```bash
# Run setup script
python3 /home/ubuntu/skills/payment-integration/scripts/setup_stripe.py

# Add Stripe keys to .env.local
STRIPE_SECRET_KEY=sk_test_...
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_...

# Use checkout template
cp templates/checkout.tsx app/components/
```

## What You Get

1. **Automated Setup** - Dependencies and configuration
2. **Checkout Component** - Ready-to-use payment button
3. **Webhook Handler** - Secure event processing
4. **Best Practices** - Security and error handling

## Supported Platforms

- Next.js (App Router, Pages Router)
- React
- Python/Flask/FastAPI

## Related Skills

- `/user-authentication-system` - Identify paying users
- `/analytics-dashboard` - Track conversion rates
- `/email-system-builder` - Send payment receipts

## Time Savings

**Manual**: 2-3 hours (SDK setup, testing, webhooks)  
**With skill**: 30 minutes  
**Saved**: ~2 hours
