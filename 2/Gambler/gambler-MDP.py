import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_single_policy(p_h, P):
	plt.xlabel("Capital")
	plt.ylabel("Bet")
	plt.title(f"Policy for p_h = {p_h}")

	Y = [ p[0] for p in P[1:-1]  ]
	X = np.arange(1, 1+len(Y))
	plt.plot(X, Y)

	plt.savefig(f"Policy-ph={p_h}.png")
	plt.close()


def display_policy(p_h, V, store_all_policies, diff_threshhold):

	cmp_threshhold = diff_threshhold
	target = len(V)-1
	P = [ [] for _ in V ]

	for capital in range(1, target):
		min_bet = 1 # Ignore 0 since it makes no progress
		max_bet = min(capital, target-capital)
			
		for bet in range(min_bet, max_bet+1):
			if np.abs(
				V[capital+bet]*p_h + V[capital-bet]*(1-p_h)- V[capital]
			) < cmp_threshhold:
				P[capital].append(bet)

		# Could not match anything due to numerical instability
		if len(P[capital]) == 0:
			best_policy = -1
			best_value = -1
			for bet in range(min_bet, max_bet+1):
				value = V[capital+bet]*p_h + V[capital-bet]*(1-p_h)
				if value > best_value:
					best_value = value
					best_policy = bet
			P[capital].append(best_policy)

	plot_single_policy(p_h, P)

	if store_all_policies:
		with open(f'Policies-ph-{p_h}.txt', 'w') as f:
			for i, p in enumerate(P[1:-1]):
				f.write(f"{i+1}: {p}\n")



def plot_value_function(p_h, V):
	plt.xlabel("Capital")
	plt.ylabel("Prob. of Success")
	plt.title(f"Value Function for p_h = {p_h}")
	Y = V[1:-1]
	X = np.arange(1, 1+len(Y))
	plt.plot(X, Y)

	plt.savefig(f"V-ph={p_h}.png")
	plt.close()


def get_value_function(p_h, n_iter, diff_threshhold, target):
	V = np.zeros(target+1)
	V[target] = 1.0

	for it in range(n_iter):
		max_update = 0
		for capital in range(1, target):
			min_bet = 0
			max_bet = min(capital, target-capital)
			
			for bet in range(min_bet, max_bet+1):
				current_capital = V[capital]

				V[capital] = max(
					current_capital,
					p_h*V[capital+bet] + (1-p_h)*V[capital-bet]
				)
				max_update = max(max_update, V[capital] - current_capital)

		if max_update < diff_threshhold:
			break

	if it == n_iter:
		print("Solution did not converge")

	return V


def main(p_h, n_iter, diff_threshhold, target, store_all_policies):
	V = get_value_function(p_h, n_iter, diff_threshhold, target)
	plot_value_function(p_h, V)
	display_policy(p_h, V, store_all_policies, diff_threshhold)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p_h', type=float, default=0.4)
    parser.add_argument('-n_iter', type=int, default=200)
    parser.add_argument('-target', type=int, default=100)
    parser.add_argument('-diff_threshhold', type=float, default=1e-6)
    parser.add_argument('--store_all_policies', action='store_true')

    return parser.parse_args()



if __name__ == '__main__':
	opt = parse_args()
	main(opt.p_h, opt.n_iter, opt.diff_threshhold, opt.target, opt.store_all_policies)