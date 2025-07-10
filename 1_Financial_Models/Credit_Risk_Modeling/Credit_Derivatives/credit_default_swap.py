import numpy as np

def price_credit_default_swap(notional, recovery_rate, hazard_rate, tenor, discount_rate):
    """
    Calculates the fair spread (or PV of legs) for a basic Credit Default Swap (CDS).
    Assumes a constant hazard rate and continuous discounting for simplicity.

    Args:
        notional (float): The notional amount of the CDS.
        recovery_rate (float): The recovery rate in case of default (e.g., 0.4 for 40%).
        hazard_rate (float): The constant hazard rate (intensity of default) per year.
        tenor (float): The maturity of the CDS in years.
        discount_rate (float): The constant risk-free discount rate per year.

    Returns:
        dict: A dictionary containing:
            - 'pv_protection_leg' (float): Present Value of the Protection Leg.
            - 'pv_premium_leg_per_bp' (float): Present Value of the Premium Leg per basis point of spread.
            - 'fair_spread_bps' (float): The fair CDS spread in basis points.
    """
    if notional <= 0 or recovery_rate < 0 or recovery_rate > 1 or hazard_rate < 0 or tenor <= 0 or discount_rate < 0:
        raise ValueError("Invalid input parameters. Check notional, recovery_rate, hazard_rate, tenor, and discount_rate.")

    # Protection Leg (Expected Loss)
    # Assuming continuous default and recovery at default
    # PV(Protection Leg) = Notional * (1 - Recovery Rate) * Integral[0 to T] (exp(-(h+r)t) * h dt)
    # Simplified for constant hazard and discount rates:
    if hazard_rate + discount_rate == 0:
        # Handle the case where the denominator is zero (e.g., both are zero)
        # This implies no default or no discounting, which simplifies the integral
        pv_protection_leg = notional * (1 - recovery_rate) * (1 - np.exp(-hazard_rate * tenor))
    else:
        pv_protection_leg = notional * (1 - recovery_rate) * hazard_rate / (hazard_rate + discount_rate) * \
                            (1 - np.exp(-(hazard_rate + discount_rate) * tenor))

    # Premium Leg (Expected Premium Payments)
    # PV(Premium Leg) = Notional * Spread * Integral[0 to T] (exp(-(h+r)t) dt)
    # Simplified for constant hazard and discount rates:
    if hazard_rate + discount_rate == 0:
        # Handle the case where the denominator is zero
        pv_premium_leg_per_unit_spread = notional * tenor
    else:
        pv_premium_leg_per_unit_spread = notional * (1 - np.exp(-(hazard_rate + discount_rate) * tenor)) / (hazard_rate + discount_rate)

    # Fair Spread (in basis points)
    # Fair Spread = PV(Protection Leg) / PV(Premium Leg per unit spread)
    if pv_premium_leg_per_unit_spread == 0:
        fair_spread = 0.0 # Or handle as infinite if protection leg is non-zero
    else:
        fair_spread = pv_protection_leg / pv_premium_leg_per_unit_spread

    fair_spread_bps = fair_spread * 10000 # Convert to basis points

    return {
        'pv_protection_leg': pv_protection_leg,
        'pv_premium_leg_per_bp': pv_premium_leg_per_unit_spread / 10000, # PV of premium leg for 1 bp spread
        'fair_spread_bps': fair_spread_bps
    }

if __name__ == '__main__':
    # Example Usage:
    N = 10_000_000  # Notional: $10 million
    RR = 0.4        # Recovery Rate: 40%
    H = 0.02        # Hazard Rate: 2% per year
    T = 5           # Tenor: 5 years
    R = 0.01        # Discount Rate: 1% per year

    cds_pricing = price_credit_default_swap(N, RR, H, T, R)

    print(f"Credit Default Swap Pricing Results:")
    print(f"  Notional: ${N:,.2f}")
    print(f"  Recovery Rate: {RR:.2%}")
    print(f"  Hazard Rate: {H:.2%}")
    print(f"  Tenor: {T} years")
    print(f"  Discount Rate: {R:.2%}")
    print(f"  PV of Protection Leg: ${cds_pricing['pv_protection_leg']:,.2f}")
    print(f"  PV of Premium Leg (per 1bp spread): ${cds_pricing['pv_premium_leg_per_bp']:,.2f}")
    print(f"  Fair CDS Spread: {cds_pricing['fair_spread_bps']:.2f} bps")

    # Example with higher hazard rate
    H_high = 0.05
    cds_pricing_high = price_credit_default_swap(N, RR, H_high, T, R)
    print(f"
Example with higher hazard rate (H={H_high:.2%}):")
    print(f"  Fair CDS Spread: {cds_pricing_high['fair_spread_bps']:.2f} bps")
