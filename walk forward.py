from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
def generate_walk_forward_sets(initial_date_str, optimization_length_months, walk_forward_length_months, num_periods,
                               end_date_str):
    # Convert the initial date and end date strings to datetime objects
    initial_date = datetime.strptime(initial_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    walk_forward_dict = []

    for i in range(num_periods):
        # Calculate the optimization period start and end dates
        opt_start_date = initial_date + relativedelta(months=walk_forward_length_months * i)
        opt_end_date = opt_start_date + relativedelta(months=optimization_length_months) - timedelta(days=1)

        # Stop if the optimization end date exceeds the end date
        if opt_end_date > end_date:
            break

        # Calculate the walk-forward period start and end dates
        wf_start_date = opt_end_date + timedelta(days=1)
        wf_end_date = wf_start_date + relativedelta(months=walk_forward_length_months) - timedelta(days=1)

        # Stop if the walk-forward end date exceeds the end date
        if wf_end_date > end_date:
            break

        # Convert dates back to string format
        optimization_period = [opt_start_date.strftime("%Y-%m-%d"), opt_end_date.strftime("%Y-%m-%d")]
        walk_forward_period = [wf_start_date.strftime("%Y-%m-%d"), wf_end_date.strftime("%Y-%m-%d")]

        # Append the periods to the list
        walk_forward_dict.append({
            'optimization_period': optimization_period,
            'walk_forward_period': walk_forward_period
        })

    return walk_forward_dict

def calculate_num_periods(start_year, end_year,optimization_length_months, walk_forward_length_months):
    start_year_number = datetime.strptime(start_year, "%Y-%m-%d").year
    end_year_number = datetime.strptime(end_year, "%Y-%m-%d").year
    end_month_number = datetime.strptime(end_year, "%Y-%m-%d").month

    # the amount of set will be calculate from this function
    num_periods = (((end_year_number - start_year_number) * 12) + (
                end_month_number - 1) - optimization_length_months) // walk_forward_length_months  # -1 to make sure that it have full month in the last set

    return num_periods


# optimization set that I think is
# 9:3
# 12:4
# 15:5
# we can simply multiply 3 months to generate the set
# we can also do the floor divisions too

# Example how to generate

# initial_date_str = "2020-01-01"
# optimization_length_months = 10
# walk_forward_length_months = 3
# end_date_str = "2023-12-31"
#
# num_periods = calculate_num_periods(start_year=initial_date_str,
#                                     end_year=end_date_str,
#                                     optimization_length_months=optimization_length_months,
#                                     walk_forward_length_months=walk_forward_length_months)
# walk_forward_sets = generate_walk_forward_sets(initial_date_str,
#                                                optimization_length_months,
#                                                walk_forward_length_months,
#                                                num_periods, end_date_str)
# for i, wf_set in enumerate(walk_forward_sets, 1):
#     print(f"Set {i}: {wf_set}")
