using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;


namespace TSP {
	class TEST {
		public static void Main(String[] Args) {
			var mySol = new Solution(Args);
			mySol.Solve();

		}
	}

	class Solution {

		void Swap<T>(ref T lhs, ref T rhs) { T t = lhs; lhs = rhs; rhs = t; }

		void InitialSetting(int n, int xmax, int ymax, long timeLimitDuration = 3000, int seed = 2525) {
			N = n;
			XMax = xmax;
			YMax = ymax;
			TimeLimitDuration = timeLimitDuration;

			X = new int[N];
			Y = new int[N];
			var used = new HashSet<ValueTuple<int, int>>();
			var rnd = new Xor128(seed);
			for (int i = 0; i < N; i++) {
				while (true) {
					var x = rnd.Next(XMax);
					var y = rnd.Next(YMax);
					if (used.Contains((x, y))) continue;
					used.Add((x, y));
					X[i] = x;
					Y[i] = y;
					break;
				}
			}
		}

		public void Solve() {
			InitialSetting(200, 100, 100, 3000, 2525);
			/* SA approach */
			SATrial();

			/* Nearest Neighbor */
			//NearestNeighborTrial();
		}

		void OutputHistory(String infotxt, String xycsv, String historycsv, double minSc, int[] minOrder) {
			using (var sw = new StreamWriter(infotxt)) {
				sw.WriteLine("N:\t{0}", N);
				sw.WriteLine("XMax:\t{0}", XMax);
				sw.WriteLine("XYax:\t{0}", YMax);
				sw.WriteLine("Duration:\t{0}", TimeLimitDuration);
				sw.WriteLine("StartTemp:\t{0}", StartTemp);
				sw.WriteLine("EmdTemp:\t{0}", EndTemp);
				sw.WriteLine("Score:\t{0}", minSc);
				sw.WriteLine("Order:\t{0}", String.Join(",", minOrder));
			}

			using (var sw = new StreamWriter(xycsv)) {
				sw.WriteLine("X,Y");
				for (int i = 0; i < N; i++) {
					sw.WriteLine("{0},{1}", X[i], Y[i]);
				}
			}
			using (var sw = new StreamWriter(historycsv)) {
				for (int i = 0; i < scores.Count; i++) {
					sw.WriteLine("{0},{1},{2}", timestamps[i], scores[i], String.Join(",", orders[i]));
				}
			}

		}
		void SATrial() {
			bool storeAccept = true;
			var (sc, ord) = SingleTrial(storeAccept);
			Console.Error.WriteLine("{0} -> {1}: {2}", StartTemp, EndTemp, sc);
			Console.WriteLine(sc);

			bool saveAcceptHistory = true;
			saveAcceptHistory &= storeAccept;
			if (saveAcceptHistory) {
				String infoPath = "info.txt";
				String cordinatePath = "cordinate.csv";
				String historyPath = "orderHistory.csv";
				OutputHistory(infoPath, cordinatePath, historyPath, sc, ord);
			}
		}

		(double, int[]) SingleTrial(bool storeAccept = false) {
			long tstart = Now();
			long timeLimitDuration = TimeLimitDuration;
			var (sc, ord) = TwoOpt(tstart, timeLimitDuration, storeAccept);
			return (sc, ord);
		}

		void NearestNeighborTrial(bool storeAccept = false) {
			var (sc, ord) = NearestNeighborMin();
			Console.Error.WriteLine("NearestNeighbor min:\t{0}", sc);

			String infoPath = "info_NN.txt";
			String xycsv = "cordinate_NN.csv";

			using (var sw = new StreamWriter(infoPath)) {
				sw.WriteLine("N:\t{0}", N);
				sw.WriteLine("XMax:\t{0}", XMax);
				sw.WriteLine("XYax:\t{0}", YMax);
				sw.WriteLine("Score:\t{0}", sc);
				sw.WriteLine("Order:\t{0}", String.Join(",", ord));
			}

			using (var sw = new StreamWriter(xycsv)) {
				sw.WriteLine("X,Y");
				for (int i = 0; i < N; i++) {
					sw.WriteLine("{0},{1}", X[i], Y[i]);
				}
			}
		}

		(double, int[]) NearestNeighborMin(bool storeAccept = false) {
			double scmin = 1e18;
			int[] ordmin = null;
			for (int i = 0; i < N; i++) {
				var (sc, ord) = NearestNeighbor(i);
				if (sc < scmin) {
					scmin = sc;
					ordmin = (int[])ord.Clone();
				}
			}
			return (scmin, ordmin);
		}
		(double, int[]) NearestNeighbor(int startnode) {
			// O(N^2)
			int[] ord = new int[N];
			ord[0] = startnode;
			bool[] used = new bool[N];
			used[startnode] = true;
			for (int i = 0; i < N - 1; i++) {
				double mi = 1e18;
				int idx = -1;
				for (int j = 0; j < N; j++) {
					if (used[j]) continue;
					if (EuclidDistance(X[ord[i]], Y[ord[i]], X[j], Y[j]) < mi) {
						mi = EuclidDistance(X[ord[i]], Y[ord[i]], X[j], Y[j]);
						idx = j;
					}
				}
				ord[i + 1] = idx;
				used[idx] = true;
			}
			var sc = Length(ord, 0, N);
			return (sc, ord);
		}

		List<int[]> orders;
		List<double> scores;
		List<long> timestamps;
		(double, int[]) TwoOpt(long tstart, long timeLimitDuration, bool storeAccept = false) {


			double minScore = (double)(XMax + YMax) * (N + 1); // infinity
			int[] minOrder = new int[N];
			Func<double, int[], bool> updateMin = (sc_, ord_) => {
				if (sc_ < minScore) {
					minScore = sc_;
					for (int j = 0; j < ord_.Length; j++) minOrder[j] = ord_[j];
					return true;
				} else {
					return false;
				}
			};

			int[] ord = Enumerable.Range(0, N).ToArray();
			var sc = Length(ord, 0, N);
			updateMin(sc, ord);

			if (storeAccept) {
				orders = new List<int[]>();
				scores = new List<double>();
				timestamps = new List<long>();

				orders.Add((int[])ord.Clone());
				scores.Add(sc);
				timestamps.Add(0);
			}

			//long tstart = Now();
			//long timeLimitDuration = TimeLimit - tstart - 200; // 200[ms] for output 
			long t1 = 0;
			long cnt = 0;
			long cnt_accept = 0;
			long cnt_update = 0;

			var rnd = new Xor128(0x5A5A5A5A);

			while (true) {
				cnt++;
				if (cnt % 1024 == 0 && (t1 = Now()) - tstart > timeLimitDuration) break;

				int l = rnd.Next(N);
				int r = rnd.Next(N);
				if (r < l) Swap(ref l, ref r);
				if (l >= r - 1) continue;
				int nl = l + 1; if (nl == N) nl = 0;
				int nr = r + 1; if (nr == N) nr = 0;

				double dl1 = EuclidDistance(X[ord[l]], Y[ord[l]], X[ord[nl]], Y[ord[nl]]);
				double dr1 = EuclidDistance(X[ord[r]], Y[ord[r]], X[ord[nr]], Y[ord[nr]]);
				double dl2 = EuclidDistance(X[ord[l]], Y[ord[l]], X[ord[r]], Y[ord[r]]);
				double dr2 = EuclidDistance(X[ord[nl]], Y[ord[nl]], X[ord[nr]], Y[ord[nr]]);

				double nsc = sc - dl1 - dr1 + dl2 + dr2;

				if (AcceptMinimizeSA(nsc, sc, t1, tstart, timeLimitDuration)) {
					cnt_accept++;
					Array.Reverse(ord, nl, r - l);
					sc = nsc;
					if (updateMin(nsc, ord)) cnt_update++;

					if (storeAccept) {
						orders.Add((int[])ord.Clone());
						scores.Add(sc);
						timestamps.Add(cnt);
					}

				}
			}

			Console.Error.WriteLine("cnt: {0}, accept: {1}, update: {2}, minScore = {3}", cnt, cnt_accept, cnt_update, minScore);
			return (minScore, minOrder);
		}

		double Length(int[] ord, int start, int length) {
			double ret = 0;
			for (int i = 0; i < length; i++) {
				int l = i, r = i + 1;
				if (r >= ord.Length) r -= ord.Length;
				ret += EuclidDistance(X[ord[l]], Y[ord[l]], X[ord[r]], Y[ord[r]]);
			}

			return ret;
		}

		double EuclidDistance(int x1, int y1, int x2, int y2) {
			x1 -= x2;
			y1 -= y2;
			return Math.Sqrt(x1 * x1 + y1 * y1);
		}

		int N;
		int XMax, YMax;
		int[] X, Y;
		long TimeLimitDuration;

		public Solution() {
		}
		public Solution(String[] args) {
			if (args.Length == 2) {
				StartTemp = double.Parse(args[0]);
				EndTemp = double.Parse(args[1]);
			}
		}

		#region SA-Template
		static Xor128 lottery = new Xor128(25252525);
		static double StartTemp = 3.69657452194861;
		static double EndTemp = 0.0000615086996743784;

		static bool AcceptMaximizeSA(double score, double prev, long now, long start, long limitDuration) {
			if (score > prev) return true;
			if (now - start >= limitDuration) return false;
			double delta = -(prev - score);
			double currentTime = (now - start) / (double)limitDuration;
			double currentTemp = StartTemp + (EndTemp - StartTemp) * currentTime;
			double acceptProb = Math.Exp(delta / currentTemp);
			return (lottery.NextD()) < acceptProb;
		}
		static bool AcceptMaximizeMC(double score, double prev, long now, long start, long limitDuration) {
			if (score > prev) return true;
			return false;
		}
		static bool AcceptMinimizeSA(double score, double prev, long now, long start, long limitDuration) {
			if (score < prev) return true;
			if (now - start >= limitDuration) return false;
			double delta = -(score - prev);
			double currentTime = (now - start) / (double)limitDuration;
			double currentTemp = StartTemp + (EndTemp - StartTemp) * currentTime;
			double acceptProb = Math.Exp(delta / currentTemp);
			double dice = lottery.NextD();
			//Console.WriteLine("sc: {0}, prev: {1}, delta:{2}, cTime: {3}, cTemp: {4}, acP: {5}, dice: {6}", score, prev, delta, currentTime, currentTemp, acceptProb, dice);
			return dice < acceptProb;
		}
		static bool AcceptMinimizeMC(double score, double prev, long now, long start, long limitDuration) {
			if (score < prev) return true;
			return false;
		}


		#endregion

		#region TimeCount

		static Stopwatch sw = Stopwatch.StartNew();//new Stopwatch();
		static bool TU = false;
		static long TimeLimit = 3000;
		static bool TimeUp() {
			if (TU) { return true; }
			return TU = (sw.ElapsedMilliseconds > TimeLimit);
		}
		static long Now() { return sw.ElapsedMilliseconds; }

		static bool TUT = false;
		static long TimeLimitT = 3000000;
		static bool TimeUpT() {
			if (TUT) { return true; }
			return TUT = (sw.ElapsedTicks > TimeLimitT);
		}
		static long NowT() { return sw.ElapsedTicks; }

		static void TimeStamp(String title) {
			Console.Error.WriteLine("{0}:{1} [ms]", title, Now());
		}
		#endregion

	}


	class Xor128 {
		uint x = 123456789;
		uint y = 362436069;
		uint z = 521288629;
		uint w = 88675123;

		public Xor128() {
		}

		public Xor128(uint seed) {
			z ^= seed;
		}
		public Xor128(int seed) {
			z ^= (uint)seed;
		}

		public uint Next() {
			uint t = x ^ (x << 11);
			x = y; y = z; z = w;
			return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
		}
		public int Next(int ul) {
			return ul == 0 ? 0 : NextI(0, ul - 1);
		}
		public int NextI(int from, int to) {
			int mod = to - from + 1;
			int ret = (int)(Next() % mod);
			return ret + from;
		}
		public double NextD() {
			return (Next() & 1048575) / 1048575.0;
		}

	}

}
