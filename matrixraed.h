#ifndef MATRIX_SUGAR_HPP
#define MATRIX_SUGAR_HPP

#include <vector>
#include <array>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <type_traits>
#include <initializer_list>
#include <algorithm>

namespace mat {

template<typename T>
struct is_numeric : std::integral_constant<bool, 
   std::is_arithmetic<typename std::remove_reference<T>::type>::value> {};

template<typename T>
struct is_matrix {
   template<typename U>
   static auto test(int) -> decltype(
       std::declval<U>().rows(),
       std::declval<U>().cols(),
       std::declval<U>()(size_t{}, size_t{}),
       std::true_type{}
   );
   template<typename>
   static std::false_type test(...);
   static constexpr bool value = decltype(test<T>(0))::value;
};

struct Slice {
   size_t start, end;
   static constexpr size_t npos = static_cast<size_t>(-1);
   Slice(size_t s = 0, size_t e = npos) : start(s), end(e) {}
   size_t size() const { return end - start; }
};

inline namespace literals {
   struct SliceBuilder {
       size_t start;
       Slice operator()(size_t end) const { return Slice(start, end); }
   };
   inline SliceBuilder operator""_sl(unsigned long long start) {
       return SliceBuilder{static_cast<size_t>(start)};
   }
}

template<typename Derived>
class ExprBase {
public:
   template<typename Other>
   auto operator+(const Other& other) const {
       return make_expr([](auto a, auto b) { return a + b; }, 
                       static_cast<const Derived&>(*this), other);
   }
   template<typename Other>
   auto operator-(const Other& other) const {
       return make_expr([](auto a, auto b) { return a - b; }, 
                       static_cast<const Derived&>(*this), other);
   }
   template<typename Other>
   auto operator*(const Other& other) const {
       return make_expr([](auto a, auto b) { return a * b; }, 
                       static_cast<const Derived&>(*this), other);
   }
   template<typename Other>
   auto operator/(const Other& other) const {
       return make_expr([](auto a, auto b) { return a / b; }, 
                       static_cast<const Derived&>(*this), other);
   }
   
   auto T() const {
       return make_unary([](auto x) { return x; }, 
                        static_cast<const Derived&>(*this), true);
   }
};

template<typename T>
class MatrixView;

template<typename T>
class Matrix : public ExprBase<Matrix<T>> {
   std::vector<T> data_;
   size_t rows_, cols_;
   
public:
   Matrix() : rows_(0), cols_(0) {}
   Matrix(size_t r, size_t c, T val = T{}) 
       : data_(r * c, val), rows_(r), cols_(c) {}
   
   Matrix(std::initializer_list<std::initializer_list<T>> init) {
       rows_ = init.size();
       cols_ = rows_ > 0 ? init.begin()->size() : 0;
       data_.reserve(rows_ * cols_);
       for (auto& row : init) {
           data_.insert(data_.end(), row.begin(), row.end());
       }
   }
   
   template<typename Expr>
   Matrix(const ExprBase<Expr>& expr) {
       const Expr& e = static_cast<const Expr&>(expr);
       rows_ = e.rows();
       cols_ = e.cols();
       data_.resize(rows_ * cols_);
       for (size_t i = 0; i < rows_; ++i)
           for (size_t j = 0; j < cols_; ++j)
               (*this)(i, j) = e(i, j);
   }
   
   T& operator()(size_t i, size_t j) { 
       return data_[i * cols_ + j]; 
   }
   const T& operator()(size_t i, size_t j) const { 
       return data_[i * cols_ + j]; 
   }
   
   MatrixView<T> operator()(Slice rs, Slice cs) {
       if (rs.end == Slice::npos) rs.end = rows_;
       if (cs.end == Slice::npos) cs.end = cols_;
       return MatrixView<T>(data_.data(), rows_, cols_, 
                          rs.start, cs.start, rs.size(), cs.size());
   }
   
   MatrixView<T> row(size_t i) {
       return (*this)(Slice(i, i+1), Slice(0, cols_));
   }
   MatrixView<T> col(size_t j) {
       return (*this)(Slice(0, rows_), Slice(j, j+1));
   }
   
   Matrix& operator=(T val) {
       std::fill(data_.begin(), data_.end(), val);
       return *this;
   }
   
   template<typename Expr>
   Matrix& operator=(const ExprBase<Expr>& expr) {
       const Expr& e = static_cast<const Expr&>(expr);
       if (rows_ != e.rows() || cols_ != e.cols()) {
           rows_ = e.rows(); cols_ = e.cols();
           data_.resize(rows_ * cols_);
       }
       for (size_t i = 0; i < rows_; ++i)
           for (size_t j = 0; j < cols_; ++j)
               (*this)(i, j) = e(i, j);
       return *this;
   }
   
   template<typename Expr>
   Matrix& operator+=(const ExprBase<Expr>& expr) {
       return *this = *this + expr;
   }
   
   size_t rows() const { return rows_; }
   size_t cols() const { return cols_; }
   
   auto begin() { return data_.begin(); }
   auto end() { return data_.end(); }
   auto begin() const { return data_.begin(); }
   auto end() const { return data_.end(); }
   
   Matrix reshape(size_t r, size_t c) const {
       Matrix res(r, c);
       res.data_ = data_;
       return res;
   }
   
   static Matrix zeros(size_t r, size_t c) { return Matrix(r, c, T{0}); }
   static Matrix ones(size_t r, size_t c) { return Matrix(r, c, T{1}); }
   static Matrix eye(size_t n) {
       Matrix m(n, n);
       for (size_t i = 0; i < n; ++i) m(i, i) = T{1};
       return m;
   }
};

template<typename T>
class MatrixView : public ExprBase<MatrixView<T>> {
   T* data_;
   size_t pr_, pc_, ro_, co_, rows_, cols_;
   
public:
   MatrixView(T* d, size_t pr, size_t pc, size_t ro, size_t co, size_t r, size_t c)
       : data_(d), pr_(pr), pc_(pc), ro_(ro), co_(co), rows_(r), cols_(c) {}
   
   T& operator()(size_t i, size_t j) { 
       return data_[(ro_ + i) * pc_ + (co_ + j)]; 
   }
   const T& operator()(size_t i, size_t j) const { 
       return data_[(ro_ + i) * pc_ + (co_ + j)]; 
   }
   
   size_t rows() const { return rows_; }
   size_t cols() const { return cols_; }
   
   template<typename Expr>
   MatrixView& operator=(const ExprBase<Expr>& expr) {
       const Expr& e = static_cast<const Expr&>(expr);
       for (size_t i = 0; i < rows_; ++i)
           for (size_t j = 0; j < cols_; ++j)
               (*this)(i, j) = e(i, j);
       return *this;
   }
   
   MatrixView& operator=(T val) {
       for (size_t i = 0; i < rows_; ++i)
           for (size_t j = 0; j < cols_; ++j)
               (*this)(i, j) = val;
       return *this;
   }
};

template<typename Op, typename LHS, typename RHS>
class BinaryExpr : public ExprBase<BinaryExpr<Op, LHS, RHS>> {
   Op op_;
   LHS lhs_;
   RHS rhs_;
   size_t rows_, cols_;
   bool transpose_;
   
public:
   BinaryExpr(Op op, const LHS& l, const RHS& r, bool t = false) 
       : op_(op), lhs_(l), rhs_(r), transpose_(t) {
       rows_ = l.rows(); 
       cols_ = l.cols();
       if (transpose_) std::swap(rows_, cols_);
   }
   
   auto operator()(size_t i, size_t j) const {
       if (transpose_) std::swap(i, j);
       return eval(i, j, typename is_matrix<RHS>::type());
   }
   
   size_t rows() const { return rows_; }
   size_t cols() const { return cols_; }
   
private:
   auto eval(size_t i, size_t j, std::true_type) const {
       return op_(lhs_(i, j), rhs_(i, j));
   }
   auto eval(size_t i, size_t j, std::false_type) const {
       return op_(lhs_(i, j), rhs_);
   }
};

template<typename Op, typename LHS>
class UnaryExpr : public ExprBase<UnaryExpr<Op, LHS>> {
   Op op_;
   LHS lhs_;
   size_t rows_, cols_;
   bool transpose_;
   
public:
   UnaryExpr(Op op, const LHS& l, bool t = false) 
       : op_(op), lhs_(l), transpose_(t) {
       rows_ = l.rows(); 
       cols_ = l.cols();
       if (transpose_) std::swap(rows_, cols_);
   }
   
   auto operator()(size_t i, size_t j) const {
       if (transpose_) std::swap(i, j);
       return op_(lhs_(i, j));
   }
   
   size_t rows() const { return rows_; }
   size_t cols() const { return cols_; }
};

template<typename Op, typename LHS, typename RHS>
BinaryExpr<Op, LHS, RHS> make_expr(Op op, const LHS& lhs, const RHS& rhs) {
   return BinaryExpr<Op, LHS, RHS>(op, lhs, rhs);
}

template<typename Op, typename LHS>
UnaryExpr<Op, LHS> make_unary(Op op, const LHS& lhs, bool t = false) {
   return UnaryExpr<Op, LHS>(op, lhs, t);
}

template<typename T>
T sum(const Matrix<T>& m) {
   return std::accumulate(m.begin(), m.end(), T{0});
}

template<typename T>
double mean(const Matrix<T>& m) {
   return static_cast<double>(sum(m)) / (m.rows() * m.cols());
}

template<typename A, typename B>
auto dot(const ExprBase<A>& a, const ExprBase<B>& b) {
   const A& ma = static_cast<const A&>(a);
   const B& mb = static_cast<const B&>(b);
   using T = decltype(ma(0,0) * mb(0,0));
   Matrix<T> res(ma.rows(), mb.cols());
   for (size_t i = 0; i < ma.rows(); ++i)
       for (size_t j = 0; j < mb.cols(); ++j)
           for (size_t k = 0; k < ma.cols(); ++k)
               res(i, j) += ma(i, k) * mb(k, j);
   return res;
}

template<typename T>
void print(const Matrix<T>& m) {
   for (size_t i = 0; i < m.rows(); ++i) {
       for (size_t j = 0; j < m.cols(); ++j)
           printf("%.4f ", static_cast<double>(m(i, j)));
       printf("\n");
   }
}

}
#endif
