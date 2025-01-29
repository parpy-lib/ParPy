pub trait SMapAccum<T: Clone> {
    fn smap_accum_l<A>(self, f: impl Fn(A, T) -> (A, T), acc: A) -> (A, Self) where Self: Sized;

    fn smap(self, f: impl Fn(T) -> T) -> Self  where Self: Sized {
        let (_, res) = self.smap_accum_l::<()>(|_, x| ((), f(x)), ());
        res
    }
}

pub trait SFold<T: Clone> {
    fn sfold<A>(&self, f: impl Fn(A, &T) -> A, acc: A) -> A where Self: Sized;
}

impl<T: Clone> SMapAccum<T> for Vec<T> {
    fn smap_accum_l<A>(self, f: impl Fn(A, T) -> (A, T), acc: A) -> (A, Vec<T>) {
        self.into_iter()
            .fold((acc, vec![]), |(acc, mut elems), x| {
                let (acc, x) = f(acc, x);
                elems.push(x);
                (acc, elems)
            })
    }
}

impl<T: Clone> SFold<T> for Vec<T> {
    fn sfold<A>(&self, f: impl Fn(A, &T) -> A, acc: A) -> A {
        self.iter().fold(acc, f)
    }
}
